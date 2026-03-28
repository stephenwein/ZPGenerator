Code Reference
==============

This page collects the main entry points of the package for day-to-day use.
ZPGenerator exposes a broad internal surface, but most workflows can start from the
high-level factories and simulation classes below.

Core User-Facing Classes
------------------------

``Pulse``
^^^^^^^^^

Imported from ``zpgenerator.time``.

Use ``Pulse`` and related time-function utilities to define shaped excitation pulses,
gates, and envelopes used by sources and detectors.

Typical uses:

* build excitation pulses for source models
* define custom detector gates
* combine simple time functions into more complex controls

``Source``
^^^^^^^^^^

Imported from ``zpgenerator.components``.

``Source`` is the main source factory. It returns pre-built source components such as:

* ``Source.two_level()``
* ``Source.purcell()``
* ``Source.phonon_assisted()``
* ``Source.exciton()``
* ``Source.biexciton()``
* ``Source.trion()``
* ``Source.fock()``
* ``Source.shaped_laser()``
* ``Source.perceval()``

Use these when you want a ready-made light source without manually assembling the lower-level component graph.

``Circuit``
^^^^^^^^^^^

Imported from ``zpgenerator.components``.

``Circuit`` is the main linear-optics factory. Common constructors include:

* ``Circuit.bs()``
* ``Circuit.ps()``
* ``Circuit.loss()``
* ``Circuit.mzi()``
* ``Circuit.perm()``
* ``Circuit.haar_random()``
* ``Circuit.from_perceval()``

Use these to build interferometers and mode transformations that can be added to a processor.

``Detector``
^^^^^^^^^^^^

Imported from ``zpgenerator.components``.

``Detector`` is the main detector factory. Common constructors include:

* ``Detector.threshold()``
* ``Detector.pnr()``
* ``Detector.vacuum()``
* ``Detector.parity()``
* ``Detector.partition()``

Use detectors to define monitored output modes and time bins in a simulated experiment.

``Processor``
^^^^^^^^^^^^^

Imported from ``zpgenerator.simulate``.

``Processor`` is the main simulation entry point. It combines sources, circuits, and detectors, then evaluates:

* detection probabilities with ``probs()``
* conditional states with ``conditional_states()``
* conditional channels with ``conditional_channels()``

It also provides convenience methods for composition, parameter updates, and several quality metrics for source characterization workflows.

Composability Layer
-------------------

``Component``
^^^^^^^^^^^^^

Imported from ``zpgenerator.network``.

``Component`` is the main low-level composition primitive. It is useful when the catalogue factories are not enough and you want to:

* assemble custom source, circuit, or detector structures
* reuse an existing subcomponent inside a larger component
* expose a custom composite object through the same processor interface

Most users should start from the catalogue factories and drop to ``Component`` only when they need custom composition.

Suggested Starting Points
-------------------------

If you are new to the package, the following documentation pages are the best practical companions to this reference:

* ``Components`` for the object model
* ``Parameters`` for runtime parameter overrides
* ``Processors`` for simulation workflows
* ``Sources``, ``Circuits``, and ``Detectors`` for the catalogue
