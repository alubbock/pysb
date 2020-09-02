from __future__ import absolute_import
from .base import SimulationResult, Simulator
from .scipyode import SerialExecutor
from functools import partial
import numpy as np
import os
if 'PYSB_JULIA_NOCM' in os.environ:
    from julia.api import Julia
    Julia(compiled_modules=False)
from julia import Main
from sympy import julia_code
import pysb.bng
import re
try:
    import diffeqpy
    from diffeqpy import de
except ImportError:
    diffeqpy = None
    de = None


class SciMLSimulator(Simulator):
    """
    SciMLSimulator using Julia's DifferentialEquations.jl from the SciML project

    .. warning::
        The interface for this class is considered experimental and may
        change without warning as PySB is updated.

    To use this module, you must first separately install Julia, which is a
    separate language from Python: https://julialang.org/downloads/

    If you receive an error when importing this class regarding Python not
    being compiled with shared library support, you can use an environment
    variable 'PYSB_JULIA_NOCM=1' to disable PyJulia compiled module support
    as an alternative to the options listed in that error message.
    The downside is this module will be slow to import.

    On first use on a given system/environment, call the
    SciMLSimulator.install() method to install the necessary Julia libraries.

    Parameters
    ----------
    model : pysb.Model
        Model to simulate.
    tspan : vector-like, optional
        Time values over which to simulate. The first and last values define
        the time range. Typically, a tuple of (start, end) should be used,
        since SciML uses adaptive time points.
    initials : vector-like or dict, optional
        Values to use for the initial condition of all species. Ordering is
        determined by the order of model.species. If not specified, initial
        conditions will be taken from model.initial_conditions (with
        initial condition parameter values taken from `param_values` if
        specified).
    param_values : vector-like or dict, optional
        Values to use for every parameter in the model. Ordering is
        determined by the order of model.parameters.
        If passed as a dictionary, keys must be parameter names.
        If not specified, parameter values will be taken directly from
        model.parameters.
    verbose : bool or int, optional (default: False)
        Sets the verbosity level of the logger. See the logging levels and
        constants from Python's logging module for interpretation of integer
        values. False is equal to the PySB default level (currently WARNING),
        True is equal to DEBUG.

    Notes
    -----
    If ``tspan`` is not defined, it may be defined in the call to the
    ``run`` method.

    Examples
    --------
    Simulate a model and display the results for an observable:

    >>> from pysb.examples.robertson import model
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> sim = SciMLSimulator(model, tspan=np.linspace(0, 40, 10))
    >>> simulation_result = sim.run()
    >>> print(simulation_result.observables['A_total']) \
        #doctest: +NORMALIZE_WHITESPACE
    [ 1.      0.899   0.8506  0.8179  0.793   0.7728  0.7557  0.7408  0.7277
    0.7158]

    For further information on retrieving trajectories (species,
    observables, expressions over time) from the ``simulation_result``
    object returned by :func:`run`, see the examples under the
    :class:`SimulationResult` class.

    """
    _supports = {'multi_initials': True,
                 'multi_param_values': True}

    def __init__(self, model, tspan=None, initials=None,
                 param_values=None, verbose=False, compiler='julia'):
        super(SciMLSimulator, self).__init__(
            model=model, tspan=tspan, initials=initials,
            param_values=param_values, verbose=verbose)

        if not diffeqpy:
            raise ImportError(
                'Error importing diffeqpy. Please install that package '
                'to continue.')

        pysb.bng.generate_equations(self._model, verbose=self.verbose)

        self._eqn_subs = {e: e.expand_expr(expand_observables=True) for
                          e in self._model.expressions}

        self._compiler = compiler
        if compiler == 'julia':
            jfn = 'function(du,u,p,t)\n'
            for i, ode in enumerate(self.model.odes):
                jfn += f'du[{i + 1}] = {julia_code(ode.subs(self._eqn_subs))}\n'
            jfn += 'end'
            jfn = self._eqn_substitutions(jfn)
            self._code_eqs = Main.eval(jfn)
        else:
            raise ValueError('Unknown compiler: {}'.format(compiler))

    @staticmethod
    def install():
        if not diffeqpy:
            raise ImportError(
                'Error importing diffeqpy. Please install that package '
                'to continue.')

        diffeqpy.install()

    def _eqn_substitutions(self, eqns):
        """String substitutions on the Julia code for the ODE RHS and
        Jacobian functions to use appropriate terms for variables and
        parameters."""
        # Substitute 'y[i+1]' for '__si'
        eqns = re.sub(r'\b__s(\d+)\b',
                      lambda m: 'u[%s]' % (int(m.group(1)) + 1),
                      eqns)

        # Substitute 'p[i]' for any named parameters
        for i, p in enumerate(self._model.parameters):
            eqns = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % (i + 1), eqns)
        return eqns

    def run(self, tspan=None, initials=None, param_values=None, **kwargs):
        """
        Run a simulation and returns the result (trajectories)

        Parameters
        ----------
        tspan
        initials
        param_values
            See parameter definitions in :class:`SciMLSimulator`.
        **kwargs: dict, optional
            Optional arguments to pass to Julia's solve() function. See
            https://diffeq.sciml.ai/dev/basics/common_solver_opts/

            A solver can be manually specified like this:

            >>> from diffeqpy import de
            >>> sim.run(alg=de.AutoTsit5(de.Rosenbrock23()))

        Returns
        -------
        A :class:`SimulationResult` object
        """
        super(SciMLSimulator, self).run(tspan=tspan,
                                        initials=initials,
                                        param_values=param_values,
                                        _run_kwargs=locals())

        # if num_processors == 1:
        #     self._logger.debug('Single processor (serial) mode')
        # else:
        #     self._logger.debug('Multi-processor (parallel) mode using {} '
        #                        'processes'.format(num_processors))

        if len(self.tspan) > 2:
            self._logger.warning(
                'SciML uses adaptive timestepping by default. '
                'Returned time points may differ from tspan.')

        with SerialExecutor() as executor:
            sim_partial = partial(_run_sciml,
                                  code_eqs=self._code_eqs,
                                  tspan=self.tspan,
                                  **kwargs)

            results = [executor.submit(sim_partial, *args)
                       for args in zip(self.initials, self.param_values)]

            try:
                trajectories, tout_all = zip(*[r.result() for r in results])
            finally:
                for r in results:
                    r.cancel()

        self._logger.info('All simulation(s) complete')
        return SimulationResult(self, tout_all, trajectories)


def _run_sciml(initials, param_values, code_eqs, tspan, **kwargs):
    prob = de.ODEProblem(code_eqs, initials,
                         (float(tspan[0]), float(tspan[-1])),
                         param_values)
    result = de.solve(prob, **kwargs)

    return np.vstack(result.u), result.t
