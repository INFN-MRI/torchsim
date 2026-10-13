"""Where a per-voxel workload runs, and how it gets there.

Simulating a volume and mapping one are both per-voxel, and both outgrow a
card long before they outgrow a host. The policy is written once, here: a
problem too small to repay a launch runs where its tensors are, and a larger
one is cut into chunks of voxels that are dealt to every CUDA device in turn,
each chunk sized so the cards hold a bounded share of it at a time, and its
answer gathered on the device the caller's tensors are on.

A workload joins by answering three questions: how much work it is, what one
voxel costs on the device, and what to do with a chunk. Choosing the devices,
sizing the chunk to the free memory, the pinned staging, and the streams that
let one chunk's transfer overlap another chunk's arithmetic are the same
whatever the kernel underneath computes.

The two policies are process-wide and are read through :func:`policy` and
:func:`offloading` rather than imported, because a caller that imported the
value would hold the one in force when it imported.
"""

from __future__ import annotations

__all__ = ["execution", "offload"]

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch

#: What streaming may hold on the devices when the caller names no budget.
_DEFAULT_BUDGET_BYTES = 512 << 20

#: Left free on every card. A recon server's product pipeline keeps its own
#: kernels and workspaces resident, and taking the last of the memory would
#: evict them rather than fail honestly.
_RESERVE_BYTES = 512 << 20


@dataclass(frozen=True)
class _Offload:
    devices: tuple[torch.device, ...]
    budget_bytes: int
    lanes: int


@dataclass(frozen=True)
class _Execution:
    """What the caller asked for, before any problem is in hand."""

    target: str  # "auto", "cpu" or "devices"
    devices: tuple[torch.device, ...]
    stream: bool | None
    budget_bytes: int | None
    lanes: int
    reserve_bytes: int


@dataclass(frozen=True)
class _Choice:
    """Where one particular problem runs: devices, chunk width and lanes."""

    devices: tuple[torch.device, ...]
    chunk: int
    lanes: int = 1


#: Whole on the host, for a problem whose tensors are elsewhere.
_ON_HOST = _Choice((torch.device("cpu"),), 1 << 62)

_DEFAULT = _Execution("auto", (), None, None, 1, _RESERVE_BYTES)
_OFFLOAD: _Offload | None = None
_EXECUTION: _Execution | None = _DEFAULT


def policy() -> _Execution | None:
    """The execution policy in force, or ``None`` if there is none."""
    return _EXECUTION


def offloading() -> _Offload | None:
    """The offload plan in force, or ``None`` if there is none."""
    return _OFFLOAD


def snapshot() -> tuple[_Execution | None, _Offload | None]:
    """The policy and offload plan in force, to be put back with :func:`restored`."""
    return _EXECUTION, _OFFLOAD


@contextmanager
def restored(state: tuple[_Execution | None, _Offload | None]) -> Iterator[None]:
    """Run inside the block under a policy taken by :func:`snapshot`.

    A backward pass runs after the call that recorded it has returned; it
    follows the policy that call ran under, not whatever is in force when the
    gradient is asked for.
    """
    global _EXECUTION, _OFFLOAD
    previous = (_EXECUTION, _OFFLOAD)
    _EXECUTION, _OFFLOAD = state
    try:
        yield
    finally:
        _EXECUTION, _OFFLOAD = previous


@contextmanager
def without_policy() -> Iterator[None]:
    """Drop the execution policy inside the block, keeping any offload plan.

    For a call that has already chosen its devices and is now running one
    shard: the choice is made, and asking again per shard would make it twice.
    """
    global _EXECUTION
    previous = _EXECUTION
    _EXECUTION = None
    try:
        yield
    finally:
        _EXECUTION = previous


@contextmanager
def unpoliced() -> Iterator[None]:
    """Run inside the block as written, whatever policy is in force.

    What a probe measures has to be the kernel, not the dispatcher's opinion
    of it.
    """
    global _EXECUTION, _OFFLOAD
    previous = (_EXECUTION, _OFFLOAD)
    _EXECUTION = None
    _OFFLOAD = None
    try:
        yield
    finally:
        _EXECUTION, _OFFLOAD = previous


@contextmanager
def offload(
    devices: Sequence[torch.device | str] = ("cuda",),
    *,
    budget_bytes: int = _DEFAULT_BUDGET_BYTES,
    lanes: int = 1,
) -> Iterator[None]:
    """Stream every per-voxel workload through CUDA devices inside the block.

    A per-voxel workload (a simulation or a mapping over many voxels) runs on
    ``devices`` in chunks of voxels sized to ``budget_bytes``, whatever its
    size and wherever its tensors are, and the result returns to the device
    the tensors came from. Transfers of one chunk overlap the arithmetic of
    another. Blocks nest; the previous plan is restored on exit.

    Parameters
    ----------
    devices : sequence of torch.device or str, default=("cuda",)
        CUDA devices to use; chunks are spread across all of them.
    budget_bytes : int, default=536870912
        Total device memory for chunk buffers, in bytes (512 MiB). Throughput
        falls steeply as it shrinks.
    lanes : int, default=1
        Chunks in flight per device. Each lane takes a share of the budget, so
        more lanes give narrower chunks. Affects throughput only.

    Raises
    ------
    ValueError
        If ``devices`` is empty or contains a non-CUDA device, or if
        ``budget_bytes`` or ``lanes`` is not positive.
    """
    global _OFFLOAD
    resolved = tuple(torch.device(value) for value in devices)
    if not resolved:
        raise ValueError("offload needs at least one device")
    for device in resolved:
        if device.type != "cuda":
            raise ValueError(f"offload is for CUDA devices, got {device}")
    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")
    if lanes <= 0:
        raise ValueError(f"lanes must be positive, got {lanes}")
    previous = _OFFLOAD
    _OFFLOAD = _Offload(resolved, int(budget_bytes), int(lanes))
    try:
        yield
    finally:
        _OFFLOAD = previous


@contextmanager
def execution(
    target: str | torch.device | Sequence[torch.device | str] = "auto",
    *,
    stream: bool | None = None,
    budget_bytes: int | None = None,
    lanes: int = 1,
    reserve_bytes: int = _RESERVE_BYTES,
) -> Iterator[None]:
    """Set where per-voxel work runs inside the block.

    With ``target="auto"``, which is also the policy outside any block, work
    too small to repay a device launch runs where its tensors are, and larger
    work is cut into chunks of voxels spread over every CUDA device. Blocks
    nest; the previous policy is restored on exit.

    Parameters
    ----------
    target : {"auto", "cpu"}, torch.device or sequence of device, default="auto"
        ``"auto"`` decides per call from the free memory of all visible CUDA
        devices. ``"cpu"`` (or CPU devices only) keeps work on the host.
        Named CUDA devices restrict the choice to them, whatever the size.
    stream : bool, default=None
        ``True`` sizes chunks to the budget alone. ``False`` gives each device
        one even share and fails when it does not fit. ``None`` takes the
        narrower of the two.
    budget_bytes : int, default=None
        Cap on device memory held while streaming, in bytes. ``None`` uses the
        smallest free memory among the devices less ``reserve_bytes``.
    lanes : int, default=1
        Chunks in flight per device, as for :func:`offload`.
    reserve_bytes : int, default=536870912
        Memory left free on each device, in bytes (512 MiB).

    Raises
    ------
    ValueError
        If the device list is empty or contains a non-CUDA device, or if
        ``budget_bytes`` or ``lanes`` is not positive.
    """
    global _EXECUTION
    if lanes <= 0:
        raise ValueError(f"lanes must be positive, got {lanes}")
    if budget_bytes is not None and budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")

    if isinstance(target, str) and target in {"auto", "cpu"}:
        name, devices = target, ()
    else:
        named = (target,) if isinstance(target, (str, torch.device)) else tuple(target)
        if not named:
            raise ValueError("execution needs at least one device")
        devices = tuple(torch.device(value) for value in named)
        if all(device.type == "cpu" for device in devices):
            name, devices = "cpu", ()
        else:
            for device in devices:
                if device.type != "cuda":
                    raise ValueError(
                        f"execution takes CUDA devices or 'cpu', got {device}"
                    )
            name = "devices"

    previous = _EXECUTION
    _EXECUTION = _Execution(
        name, devices, stream, budget_bytes, int(lanes), int(reserve_bytes)
    )
    try:
        yield
    finally:
        _EXECUTION = previous


def _free_bytes(device: torch.device, reserve: int) -> int:
    """What this card can spare, after leaving the reserve alone."""
    free, _total = torch.cuda.mem_get_info(device)
    return max(0, free - reserve)


def _indexed(device: torch.device) -> torch.device:
    """``device`` with its index named, so the same card compares equal."""
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _candidate_devices(plan: _Execution) -> tuple[torch.device, ...]:
    if plan.devices:
        return tuple(_indexed(device) for device in plan.devices)
    return tuple(
        torch.device("cuda", index) for index in range(torch.cuda.device_count())
    )


#: Work below which a per-voxel body does not repay a device launch. It came
#: off one laptop GPU, so it is a place to start rather than a description of
#: any particular machine; every mapping worth running is far past it.
PER_VOXEL_CROSSOVER = 2_000_000.0


def chunk_voxels(plan: _Offload, bytes_per_voxel: int) -> int:
    """Largest voxel chunk whose buffers fit the budget, across every lane."""
    across = plan.lanes * len(plan.devices)
    return max(1, plan.budget_bytes // (across * max(1, bytes_per_voxel)))


def choose(
    *,
    work: int,
    voxels: int,
    bytes_per_voxel: int,
    crossover: float,
    home: torch.device,
) -> _Choice | None:
    """Decide where one per-voxel problem runs, given the policy in force.

    Under ``"auto"`` a problem below the crossover runs where its tensors are.
    Past it, the voxels are cut into chunks and dealt to every CUDA device in
    turn: a chunk is as wide as the smallest card's free memory allows, and no
    wider than an even share, so every card has work.

    Parameters
    ----------
    work:
        How much arithmetic the problem is, in the unit ``crossover``
        answers in.
    voxels:
        The independent axis, which is what a chunk is a slice of.
    bytes_per_voxel:
        Device memory one voxel needs for this pass.
    crossover:
        The work below which a launch does not repay itself.
    home:
        The device the problem's tensors are on, where the answer goes back.

    Returns
    -------
    _Choice or None
        ``None`` means the call stands as written: it runs whole wherever its
        tensors already are.
    """
    if voxels == 0:
        return None
    forced = _OFFLOAD
    if forced is not None:
        chunk = min(voxels, chunk_voxels(forced, bytes_per_voxel))
        devices = tuple(_indexed(device) for device in forced.devices)
        return _Choice(devices, max(1, chunk), forced.lanes)
    plan = _EXECUTION
    if plan is None:
        return None
    if plan.target == "cpu" or not torch.cuda.is_available():
        return None if home.type == "cpu" else _ON_HOST
    devices = _candidate_devices(plan)
    if not devices:
        return None if home.type == "cpu" else _ON_HOST
    if plan.target == "auto" and work < crossover:
        return None

    share = -(-voxels // len(devices))
    if plan.stream is False:
        chunk = share
    else:
        if plan.budget_bytes is not None:
            budget = plan.budget_bytes // len(devices)
        else:
            budget = min(_free_bytes(device, plan.reserve_bytes) for device in devices)
        fits = max(1, budget // (plan.lanes * max(1, bytes_per_voxel)))
        chunk = min(voxels, fits) if plan.stream else min(share, fits)
    if len(devices) == 1 and devices[0] == home and chunk >= voxels:
        return None
    return _Choice(devices, max(1, chunk), plan.lanes)


def one_device(
    *,
    work: int,
    voxels: int,
    bytes_per_voxel: int,
    crossover: float,
    home: torch.device,
) -> torch.device | None:
    """Where a chunked reduction should run, under the policy in force.

    A reduction keeps small accumulators in one place and feeds them a chunk
    at a time, so unlike a per-voxel map it wants a device rather than a plan:
    the chunking is already the caller's, and what crosses is the chunk. The
    arguments mean what they mean for :func:`choose`.

    Returns
    -------
    torch.device or None
        ``None`` means the caller should run where its tensors already are.
    """
    choice = choose(
        work=work,
        voxels=voxels,
        bytes_per_voxel=bytes_per_voxel,
        crossover=crossover,
        home=home,
    )
    if choice is None:
        return None
    return choice.devices[0]


def cut(
    value: torch.Tensor, axis: int, begin: int, end: int, device: torch.device
) -> torch.Tensor:
    """One chunk of ``value`` along its voxel axis, contiguous on ``device``.

    A chunk leaving the host goes through pinned memory, so its copy runs on
    the chunk's stream and overlaps whatever the device is already doing.
    """
    piece = value.narrow(axis, begin, end - begin).contiguous()
    if piece.device == device:
        return piece
    if piece.device.type == "cpu" and device.type == "cuda":
        return piece.pin_memory().to(device, non_blocking=True)
    if device.type == "cpu":
        return piece.to(device)
    return piece.to(device, non_blocking=True)


def in_chunks(
    choice: _Choice,
    voxels: int,
    home: torch.device,
    pieces: Callable[[int, int, torch.device], Any],
    body: Callable[[Any, torch.device], Sequence[torch.Tensor]],
    axes: Callable[[int, torch.Tensor, int], int | None],
) -> tuple[torch.Tensor, ...]:
    """Run a per-voxel body a chunk at a time and assemble it on ``home``.

    Chunks are dealt to the devices' lanes in turn, each lane on a stream of
    its own, so one chunk's transfers overlap another's arithmetic and every
    card computes at once. Nothing in the loop reads a device result, which is
    what lets the host run ahead and keep the lanes fed.

    Parameters
    ----------
    choice:
        The devices, chunk width and lanes from :func:`choose`.
    voxels:
        Length of the voxel axis being chunked.
    home:
        Where the assembled answers go.
    pieces:
        ``pieces(begin, end, device)`` returns the chunk's inputs on
        ``device``.
    body:
        ``body(inputs, device)`` returns the chunk's answers.
    axes:
        ``axes(index, answer, width)`` names the voxel axis of the
        ``index``-th answer of a chunk ``width`` voxels wide, or ``None`` for
        an answer every chunk contributes to, which is summed.

    Returns
    -------
    tuple of torch.Tensor
        The answers, on ``home``.
    """
    # The answers are allocated on the stream the caller queues on, which the
    # lanes' own streams are not, so the caller can use them once the loop ends.
    allocating = (
        torch.cuda.stream(torch.cuda.current_stream(home))
        if home.type == "cuda"
        else nullcontext()
    )
    lanes = [_Lane(device) for device in choice.devices for _ in range(choice.lanes)]
    width = choice.chunk
    placed: dict[int, tuple[int | None, torch.Tensor | list[torch.Tensor]]] = {}
    count = 0
    for index, begin in enumerate(range(0, voxels, width)):
        end = min(voxels, begin + width)
        lane = lanes[index % len(lanes)]
        with lane.running(), unpoliced():
            answers = body(pieces(begin, end, lane.device), lane.device)
            count = len(answers)
            for position, answer in enumerate(answers):
                if position not in placed:
                    axis = axes(position, answer, end - begin)
                    if axis is None:
                        placed[position] = (None, [])
                    else:
                        shape = list(answer.shape)
                        shape[axis] = voxels
                        with allocating:
                            whole = torch.empty(
                                shape,
                                dtype=answer.dtype,
                                device=home,
                                pin_memory=home.type == "cpu"
                                and lane.device.type == "cuda",
                            )
                        placed[position] = (axis, whole)
                axis, held = placed[position]
                if axis is None:
                    held.append(answer)
                else:
                    held.narrow(axis, begin, end - begin).copy_(
                        answer, non_blocking=True
                    )
            # Dropped before the next chunk is cut, so a lane holds one chunk.
            del answers, answer
    for lane in lanes:
        lane.finish()
    answers = []
    for position in range(count):
        axis, held = placed[position]
        if axis is None:
            total = held[0].to(home)
            for answer in held[1:]:
                total = total + answer.to(home)
            answers.append(total)
        else:
            answers.append(held)
    return tuple(answers)


class _Lane:
    """One stream on one device, which the chunks dealt to it are queued on."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.stream = None
        if device.type == "cuda":
            self.stream = torch.cuda.Stream(device=device)
            # Whatever the caller queued before the loop -- the inputs among
            # it -- is finished before this lane reads anything.
            self.stream.wait_stream(torch.cuda.current_stream(device))

    @contextmanager
    def running(self) -> Iterator[None]:
        """Queue on this lane's stream inside the block."""
        if self.stream is None:
            yield
            return
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            yield

    def finish(self) -> None:
        """Make the device's own stream wait for this lane, then the host."""
        if self.stream is None:
            return
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        self.stream.synchronize()


def per_voxel(
    inputs: Sequence[torch.Tensor],
    *,
    bytes_per_voxel: int,
    work: int,
    crossover: float,
    body: Callable[[Sequence[torch.Tensor], torch.device], tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, ...] | None:
    """Run a per-voxel body wherever the policy in force says to.

    Voxels are independent, so the only question is how many go across at
    once and to which card. ``body`` is handed one chunk of every input,
    already on a device, and returns that chunk's answer.

    Parameters
    ----------
    inputs:
        Tensors sharing a leading voxel axis, which is the axis chunked.
    bytes_per_voxel:
        What one voxel needs on the device, for sizing a chunk against the
        memory budget.
    work:
        The problem's size, in the unit ``crossover`` answers in.
    crossover:
        The work below which a launch does not repay itself.
    body:
        Called as ``body(chunk, device)``, returning one or more tensors that
        share the chunk's voxel axis.

    Returns
    -------
    tuple of torch.Tensor, or None
        The assembled answers, on the device the inputs came from. ``None``
        means the caller should proceed as written.
    """
    voxels = int(inputs[0].shape[0])
    home = inputs[0].device
    choice = choose(
        work=work,
        voxels=voxels,
        bytes_per_voxel=bytes_per_voxel,
        crossover=crossover,
        home=home,
    )
    if choice is None:
        return None
    return in_chunks(
        choice,
        voxels,
        home,
        lambda begin, end, device: [
            cut(value, 0, begin, end, device) for value in inputs
        ],
        body,
        lambda _position, _answer, _width: 0,
    )
