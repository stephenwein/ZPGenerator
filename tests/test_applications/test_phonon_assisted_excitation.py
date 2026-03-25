from math import isclose
from numpy import pi

from zpgenerator import Material, Pulse, Source


def _make_source():
    return Source.phonon_assisted(
        pulse=Pulse.gaussian({'width': 10, 'area': 20 * pi}),
        parameters={'emitter/resonance': -0.5, 'emitter/dephasing': 0.01},
        purcell_factor=10,
        regime=0.1,
        timescale=200,
        temperature=7,
        material=Material.ingaas_quantum_dot(),
    )


def test_phonon_assisted_uses_emitter_scoped_pulse_parameters():
    source = _make_source()

    assert 'detuning' not in source.parameters
    assert 'emitter/detuning' not in source.parameters
    assert 'emitter/phase' in source.parameters


def test_phonon_assisted_brightness_and_g2_regression():
    source = _make_source()

    assert isclose(source.mu(), 0.57105, abs_tol=5e-3)
    assert isclose(source.mu(parameters={'emitter/resonance': 0}), 0.46942, abs_tol=5e-3)

    assert isclose(source.g2(), 0.11459, abs_tol=5e-3)
    assert isclose(source.g2(parameters={'emitter/resonance': 0}), 0.26169, abs_tol=5e-3)


def test_phonon_assisted_cavity_detuning_changes_the_emission_statistics():
    source = _make_source()

    detuned_mu = source.mu(parameters={'cavity/resonance': -0.5})
    detuned_g2 = source.g2(parameters={'cavity/resonance': -0.5})

    assert detuned_mu < 1
    assert detuned_g2 < 1
    assert detuned_mu != source.mu()
