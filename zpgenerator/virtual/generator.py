from ..system import AElement
from ..network import Component, AComponent
from .propagator import VPropNHTD, VPropHTD, VPropTI


class Generator:
    """a propagator factory that chooses the right propagator for a given time step"""

    def __init__(self, component: AElement, binned_detectors: dict = None,
                 lifetime_mode: int = None, precision: int = 6):
        self.component = component if isinstance(component, AComponent) else Component(component)
        # Keep virtual-tree trajectories unnormalised; these traces encode generating points.
        self.default_options = {"nsteps": 500000,
                                "atol": 10 ** -precision,
                                "rtol": 10 ** -(precision),
                                "normalize_output": False}
        self.lifetime_mode = lifetime_mode
        self.binned_detectors = self.component.output.binned_detectors if binned_detectors is None else binned_detectors

    def build_propagator(self, t: float, parameters: dict = None, options: dict = None):
        options = self.default_options if options is None else options
        if isinstance(options, dict):
            options = dict(options)
            options["normalize_output"] = False
        elif hasattr(options, "normalize_output"):
            options.normalize_output = False

        use_fourier_nhtd = self._requires_fourier_virtual_configs() and not self._prefer_htd_for_parameters(parameters)
        parameters = self._canonicalize_fourier_parameters(parameters) if use_fourier_nhtd else parameters

        quadruple = self.component.evaluate_quadruple(t, parameters)
        hamiltonian = quadruple.hamiltonian
        environment = quadruple.environment
        transitions = quadruple.transitions

        population = transitions[self.lifetime_mode].num() if self.lifetime_mode is not None else None
        if population:
            expect_operator = [population.constant] if not population.variable \
                else [lambda t, rho_t: (population.evaluate(t) * rho_t).tr()]
        else:
            expect_operator = None

        jumps = [sum(time_bin.detector.coupling_function(t, self.component.set_parameters(parameters)) *
                     transitions[time_bin.mode].jump()
                     for time_bin in time_bins) for time_bins in self.binned_detectors.values()]

        if use_fourier_nhtd:
            generator = hamiltonian.liou() + sum(env.lind() if not env.is_super else env for env in environment)
            return VPropNHTD(generator=generator,
                            jumps=jumps,
                            expect_operators=expect_operator,
                            options=options)

        if self.component.is_time_dependent(t, parameters) or population is not None:
            if self.component.is_nonhermitian_time_dependent(t, parameters):
                generator = hamiltonian.liou() + sum(env.lind() if not env.is_super else env for env in environment)
                return VPropNHTD(generator=generator,
                                 jumps=jumps,
                                 expect_operators=expect_operator,
                                 options=options)
            else:
                return VPropHTD(hamiltonian=hamiltonian.list_form(),
                                collapse_operators=[env.list_form() for env in environment],
                                jumps=[jump.constant for jump in jumps],
                                expect_operators=expect_operator,
                                options=options)
        else:
            #  Add TI method eventually
            return VPropHTD(hamiltonian=hamiltonian.list_form(),
                            collapse_operators=[env.list_form() for env in environment],
                            jumps=[jump.constant for jump in jumps],
                            expect_operators=expect_operator,
                            options=options)

    def _requires_fourier_virtual_configs(self) -> bool:
        for time_bins in self.binned_detectors.values():
            for time_bin in time_bins:
                detector = time_bin.detector
                method = getattr(detector, "method", "Threshold")
                resolution = getattr(detector, "resolution", None)
                if method == "Fourier":
                    return True
                if method == "Threshold" and resolution not in (None, 0, 1):
                    return True
        return False

    @staticmethod
    def _prefer_htd_for_parameters(parameters: dict = None) -> bool:
        if not parameters:
            return False
        detuning = {}
        resonance = {}
        for key, value in parameters.items():
            if not isinstance(key, str):
                continue
            if key == "detuning" or key.endswith("/detuning"):
                prefix = key[:-len("/detuning")] if key.endswith("/detuning") else ""
                detuning[prefix] = value
            elif key == "resonance" or key.endswith("/resonance"):
                prefix = key[:-len("/resonance")] if key.endswith("/resonance") else ""
                resonance[prefix] = value

        if not detuning and not resonance:
            return False

        # If users explicitly set matching detuning and resonance for the same scope,
        # the effective rotating-frame shift is unchanged; keep the NHTD/Fourier path.
        shared = set(detuning).intersection(resonance)
        if shared and all(detuning[p] == resonance[p] for p in shared):
            if set(detuning) == shared and set(resonance) == shared:
                return False

        return True

    @staticmethod
    def _canonicalize_fourier_parameters(parameters: dict = None) -> dict:
        if not parameters:
            return parameters
        updated = dict(parameters)
        detuning_keys = [k for k in updated.keys() if isinstance(k, str) and (k == "detuning" or k.endswith("/detuning"))]
        for det_key in detuning_keys:
            prefix = det_key[:-len("/detuning")] if det_key.endswith("/detuning") else ""
            res_key = f"{prefix}/resonance" if prefix else "resonance"
            if res_key in updated and updated[res_key] == updated[det_key]:
                updated[det_key] = 0
                updated[res_key] = 0
        return updated
