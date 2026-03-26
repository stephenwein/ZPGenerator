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


def _make_notebook_source(open_all: bool = False):
    source = Source.phonon_assisted(
        pulse=Pulse.gaussian(parameters={'width': 10 / (4 * 0.6931471805599453) ** 0.5}),
        purcell_factor=10,
        regime=0.1,
        timescale=110,
        temperature=7,
        material=Material.ingaas_quantum_dot(),
    )
    if open_all:
        source.output.open_all()
    return source


def _resonance_scan_parameters(detuning: float, parameters: dict = None):
    return {
        'emitter/resonance': -detuning,
        'cavity/resonance': -detuning,
        **(parameters or {}),
    }


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


def test_phonon_assisted_notebook_sideband_is_comparable_to_resonance():
    source = _make_notebook_source()

    resonant_mu = source.mu(parameters=_resonance_scan_parameters(0, {'emitter/area': pi}))
    sideband_mu = source.mu(parameters=_resonance_scan_parameters(0.6, {'emitter/area': 14.25 * pi}))

    assert isclose(resonant_mu, 0.90236, abs_tol=1e-2)
    assert isclose(sideband_mu, 0.72645, abs_tol=1e-2)
    assert sideband_mu / resonant_mu > 0.75


def test_phonon_assisted_notebook_cavity_port_sideband_outperforms_resonance_at_high_area():
    source = _make_notebook_source(open_all=True)

    resonant = source.photon_statistics(
        port=1,
        truncation=2,
        parameters=_resonance_scan_parameters(0, {'emitter/area': 20 * pi}),
    )
    sideband = source.photon_statistics(
        port=1,
        truncation=2,
        parameters=_resonance_scan_parameters(0.6, {'emitter/area': 20 * pi}),
    )

    assert sideband.beta() > resonant.beta()
    assert sideband.g2() < resonant.g2()
