from math import isclose

from numpy import real

from zpgenerator import Circuit, Detector, Processor, Source


def test_photonic_circuits_notebook_explicit_efficiency_paths():
    qpu = Processor()
    qpu.add(
        [0, 1],
        Source.perceval(
            emission_probability=0.85,
            multiphoton_component=0.05,
            indistinguishability=0.9,
            name="source",
        ),
    )
    qpu.add(0, Circuit.bs())
    qpu.add([0, 1], Detector.pnr(4, name="detector"))

    probs_source_only = qpu.probs(parameters={"source/efficiency": 0.5})
    probs_both = qpu.probs(parameters={"source/efficiency": 0.5, "detector/efficiency": 0.5})

    assert probs_source_only != probs_both
    assert isclose(probs_source_only[1, 1], 0.014142060024640033, abs_tol=1e-6)
    assert isclose(probs_both[1, 1], 0.004064001059141389, abs_tol=1e-6)


def test_photonic_circuits_notebook_lossy_mzi_endpoints():
    qpu = (
        Processor()
        // Source.fock(1)
        // Circuit.bs()
        // Circuit.loss(name="arm 0")
        // Circuit.bs()
        // ([0, 1], Detector.threshold())
    )

    probs_lossy = qpu.probs(parameters={"arm 0/efficiency": 0})
    assert isclose(probs_lossy[0, 0], 0.5, abs_tol=1e-6)
    assert isclose(probs_lossy[0, 1], 0.25, abs_tol=1e-6)
    assert isclose(probs_lossy[1, 0], 0.25, abs_tol=1e-6)

    probs_ideal = qpu.probs(parameters={"arm 0/efficiency": 1.0})
    assert isclose(probs_ideal[0, 1], 1.0, abs_tol=1e-6)
    assert isclose(probs_ideal[1, 0], 0.0, abs_tol=1e-6)


def test_wigner_notebook_parity_detector_for_fock_sequence():
    expected = {1: -1, 2: 1, 3: -1, 4: 1, 5: -1}

    for photon_number, parity in expected.items():
        qpu = Processor() // Source.fock(photon_number) // Detector.parity()
        probs = qpu.probs()

        assert isclose(real(probs["p"]), parity, abs_tol=1e-5)
