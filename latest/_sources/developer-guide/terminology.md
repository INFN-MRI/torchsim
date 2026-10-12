# Terminology

The terms and conventions the code and these pages use.

## Signal model

A **signal model** is a {class}`~blochsim.Simulator` subclass: a handler
per kind of event, and a layout of one repetition
({doc}`../explanations/signal-model`). It is the only thing a sequence has to
supply; differentiation, device placement, parameter estimation, model-based
reconstruction and sequence design are written once against that interface.

(dev-units)=

## Units

Units are public at the edges and internal underneath. A caller writes
milliseconds, degrees and Hz; a description timestamps in microseconds and
carries radians. Convert at the boundary, and name the unit in the identifier
(`duration_s`, `flip_rad`, `t1_ms`) rather than in a comment beside it.

## Course and Applications

The gallery has two kinds of page. A **Course lesson** is one of the five
numbered pages read in order, from a shipped simulator to a signal model and
an operator of your own. An **Application** is a complete worked example built
on the Course, which stands alone and names the lessons it assumes.
