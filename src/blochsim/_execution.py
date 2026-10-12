"""Where a per-voxel workload runs, and how it gets there.

Simulating a volume and mapping one are both per-voxel, and both outgrow a
card long before they outgrow a host. What to do about that is one policy --
stay on the host when a launch would not repay itself, cross whole when the
problem fits, stream through in chunks when it does not, and spread across as
many cards as there are -- and it is written once, here.

A workload joins by answering three questions: how much work it is, what one
voxel costs on the device, and what to do with a chunk. Choosing the devices,
sizing the chunk to a memory budget, the pinned staging, and the streams that
let one chunk's transfer overlap another chunk's arithmetic are the same
whatever the kernel underneath computes.

The two policies are process-wide and are read through :func:`policy` and
:func:`offloading` rather than imported, because a caller that imported the
value would hold the one in force when it imported.
"""

from __future__ import annotations

__all__ = ["execution", "offload"]

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
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
    """What to do with one particular problem."""

    where: str  # "cpu", "upfront" or "stream"
    devices: tuple[torch.device, ...] = ()
    offload: _Offload | None = None


_OFFLOAD: _Offload | None = None
_EXECUTION: _Execution | None = None
_ON_HOST = _Choice("cpu")


def policy() -> _Execution | None:
    """The execution policy in force, or ``None`` if there is none."""
    return _EXECUTION


def offloading() -> _Offload | None:
    """The offload plan in force, or ``None`` if there is none."""
    return _OFFLOAD


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
    """Context manager that streams host-resident volumes through CUDA devices.

    Inside the block, a workload whose voxels are on the host runs on
    ``devices`` in chunks, and the result returns on the host.

    Parameters
    ----------
    devices : sequence of torch.device or str, optional
        CUDA devices to use. Defaults to ``("cuda",)``.
    budget_bytes : int, optional
        Total device memory for chunk buffers, in bytes. Throughput falls
        steeply as it shrinks. Defaults to 512 MiB.
    lanes : int, optional
        Chunks in flight per device. Each lane takes a share of the budget, so
        more lanes give narrower chunks. Affects throughput only. Defaults
        to 1.

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
    """Context manager that sets where per-voxel work runs.

    With ``"auto"``, work too small to repay a launch stays on the CPU, work
    that fits on the devices is sent whole, and larger work is streamed in
    chunks. Outside a block, work runs where its tensors are.

    Parameters
    ----------
    target : {"auto", "cpu"}, torch.device or sequence of device, optional
        ``"auto"`` decides per call from the free memory of the CUDA devices;
        otherwise the host or the named CUDA devices are used. Defaults to
        ``"auto"``.
    stream : bool, optional
        ``True`` always streams in chunks; ``False`` requires the whole volume
        to be resident, and fails if it does not fit. Default decides per call.
    budget_bytes : int, optional
        Cap on device memory held while streaming, in bytes. Defaults to the
        free memory less ``reserve_bytes``.
    lanes : int, optional
        Chunks in flight per device, as for :func:`offload`. Defaults to 1.
    reserve_bytes : int, optional
        Memory left free on each device, in bytes. Defaults to 512 MiB.

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


def _candidate_devices(plan: _Execution) -> tuple[torch.device, ...]:
    if plan.devices:
        return plan.devices
    return tuple(
        torch.device("cuda", index) for index in range(torch.cuda.device_count())
    )


#: Work below which a per-voxel body does not repay a device launch. It came
#: off one laptop GPU, so it is a place to start rather than a description of
#: any particular machine; every mapping worth running is far past it.
PER_VOXEL_CROSSOVER = 2_000_000.0


def choose(
    *,
    work: int,
    voxels: int,
    bytes_per_voxel: int,
    crossover: float,
) -> _Choice | None:
    """Decide where one per-voxel problem runs, given the policy in force.

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

    Returns
    -------
    _Choice or None
        ``None`` means no policy is in force and the call stands as written:
        it runs wherever its tensors already are.
    """
    plan = _EXECUTION
    if plan is None:
        return None
    if plan.target == "cpu":
        return _ON_HOST
    if not torch.cuda.is_available():
        return _ON_HOST

    devices = _candidate_devices(plan)
    if not devices:
        return _ON_HOST

    if plan.target == "auto" and work < crossover:
        return _ON_HOST

    footprint = voxels * bytes_per_voxel
    free = [_free_bytes(device, plan.reserve_bytes) for device in devices]

    if plan.stream is not True:
        # Fewest cards that can hold it, so the rest stay out of the way.
        for count in range(1, len(devices) + 1):
            if sum(free[:count]) >= footprint:
                return _Choice("upfront", devices[:count])
        if plan.stream is False:
            # Asked for resident and it will not fit; let the allocator say so
            # rather than silently doing something else.
            return _Choice("upfront", devices)

    budget = plan.budget_bytes or max(1, min(free) if free else 0)
    return _Choice("stream", devices, _Offload(devices, max(1, budget), plan.lanes))


def chunk_voxels(plan: _Offload, bytes_per_voxel: int) -> int:
    """Largest voxel chunk whose buffers fit the budget, across every lane."""
    across = plan.lanes * len(plan.devices)
    return max(1, plan.budget_bytes // (across * max(1, bytes_per_voxel)))


class Lane:
    """One device stream, and the pinned buffer its transfers go through."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.drained = torch.cuda.Event()
        # Claimed by the first :meth:`stage`; a pass that only brings results
        # home never needs it.
        self.staging: torch.Tensor | None = None

    def upload(self, piece: torch.Tensor) -> torch.Tensor:
        """One chunk on this lane's device.

        A chunk coming from the host goes through pinned memory so the copy
        can overlap another chunk's arithmetic; one already on a device is
        left where it is.
        """
        if piece.device == self.device:
            return piece
        if piece.device.type != "cpu":
            return piece.to(self.device)
        landed = torch.empty(piece.shape, dtype=piece.dtype, device=self.device)
        return self.stage(piece, landed)

    def stage(self, piece: torch.Tensor, landed: torch.Tensor) -> torch.Tensor:
        """Copy one host chunk into a device buffer, and return it there.

        The staging buffer is pinned, so the transfer runs asynchronously and
        the stream reads the buffer long after the host wrote it. Overwriting
        it for the next chunk therefore has to wait for that read.
        """
        size = piece.numel()
        if (
            self.staging is None
            or self.staging.numel() < size
            or self.staging.dtype != piece.dtype
        ):
            self.staging = torch.empty(
                size, dtype=piece.dtype, device="cpu", pin_memory=True
            )
        self.drained.synchronize()
        host = self.staging[:size].view(piece.shape)
        host.copy_(piece)
        landed.copy_(host, non_blocking=True)
        self.drained.record(torch.cuda.current_stream(self.device))
        return landed


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
        means no policy is in force and the caller should proceed as written.
    """
    voxels = int(inputs[0].shape[0])
    choice = choose(
        work=work,
        voxels=voxels,
        bytes_per_voxel=bytes_per_voxel,
        crossover=crossover,
    )
    if choice is None:
        return None
    home = inputs[0].device
    if choice.where == "cpu":
        gathered = body([value.cpu() for value in inputs], torch.device("cpu"))
    elif choice.where == "upfront":
        gathered = _spread(inputs, voxels, choice.devices, body)
    else:
        gathered = _streamed(inputs, voxels, choice.offload, bytes_per_voxel, body)
    return tuple(value.to(home) for value in gathered)


def one_device(
    *,
    work: int,
    voxels: int,
    bytes_per_voxel: int,
    crossover: float,
) -> torch.device | None:
    """Where a chunked reduction should run, under the policy in force.

    A reduction keeps small accumulators in one place and feeds them a chunk
    at a time, so unlike a per-voxel map it wants a device rather than a plan:
    the chunking is already the caller's, and what crosses is the chunk. The
    arguments mean what they mean for :func:`per_voxel`.

    Returns
    -------
    torch.device or None
        ``None`` means no policy is in force and the caller should run where
        its tensors already are.
    """
    choice = choose(
        work=work,
        voxels=voxels,
        bytes_per_voxel=bytes_per_voxel,
        crossover=crossover,
    )
    if choice is None:
        return None
    if choice.where == "cpu":
        return torch.device("cpu")
    return choice.devices[0]


def _spread(
    inputs: Sequence[torch.Tensor],
    voxels: int,
    devices: tuple[torch.device, ...],
    body: Callable[[Sequence[torch.Tensor], torch.device], tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, ...]:
    """One share of the voxels to each card, resident, then joined."""
    share = -(-voxels // len(devices))
    pieces = []
    for index, device in enumerate(devices):
        begin = index * share
        end = min(voxels, begin + share)
        if begin >= end:
            break
        answer = body([value[begin:end].to(device) for value in inputs], device)
        pieces.append(tuple(value.cpu() for value in answer))
    return _joined(pieces)


def _streamed(
    inputs: Sequence[torch.Tensor],
    voxels: int,
    plan: _Offload,
    bytes_per_voxel: int,
    body: Callable[[Sequence[torch.Tensor], torch.device], tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, ...]:
    """The volume through the cards a chunk at a time, transfers overlapping."""
    chunk = max(1, min(voxels, chunk_voxels(plan, bytes_per_voxel)))
    lanes = [Lane(device) for device in plan.devices for _ in range(plan.lanes)]
    pieces: list[Any] = [None] * ((voxels + chunk - 1) // chunk)

    def run(lane: Lane, begin: int, end: int) -> None:
        moved = [lane.upload(value[begin:end]) for value in inputs]
        answer = body(moved, lane.device)
        pieces[begin // chunk] = tuple(value.cpu() for value in answer)

    stream_chunks(plan, voxels, chunk, lanes, run)
    return _joined([piece for piece in pieces if piece is not None])


def _joined(pieces: Sequence[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    """Each chunk's answers, concatenated along the voxel axis."""
    return tuple(torch.cat(column, dim=0) for column in zip(*pieces, strict=True))


def stream_chunks(
    plan: _Offload,
    voxels: int,
    chunk: int,
    lanes: Sequence[Any],
    body: Callable[[Any, int, int], None],
) -> None:
    """Walk the volume one chunk per lane, then wait for every stream.

    The launches are asynchronous, so the loop runs ahead of the devices and a
    chunk's transfer can proceed while another chunk computes.
    """
    for index, begin in enumerate(range(0, voxels, chunk)):
        end = min(voxels, begin + chunk)
        lane = lanes[index % len(lanes)]
        with torch.cuda.stream(lane.stream):
            body(lane, begin, end)
    for lane in lanes:
        torch.cuda.current_stream(lane.device).wait_stream(lane.stream)
    for device in plan.devices:
        torch.cuda.synchronize(device)
