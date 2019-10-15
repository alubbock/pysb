from pysb.experimental.newmodel import Monomer, Parameter, Expression, Rule, \
    Initial, Observable, MultiState, Model
from pysb.simulator import ScipyOdeSimulator
from pysb.export.pysb_flat import PysbFlatExporter
from unittest import TestCase
import copy
from pysb.examples import earm_1_3, bngwiki_egfr_simple


class TestNewModelClass(TestCase):
    def setUp(self):
        class MyModel(Model):
            A = Monomer()
            k1 = Parameter(2)
            A_0 = Parameter(5)
            e1 = Expression(k1 * 2)
            r1 = Rule(None >> A(), k1)
            initials = [Initial(A(), k1), ]
            o1 = Observable(A())

        self.Model = MyModel
        self.model = MyModel()

    def test_deepcopy_instance(self):
        for c in self.model.components:
            assert c.name is not None

        old = PysbFlatExporter(self.model).export()
        cpy = copy.deepcopy(self.model)
        new = PysbFlatExporter(cpy).export()
        assert new == old

    def test_export(self):
        PysbFlatExporter(self.model).export()

    def test_simulate_ode(self):
        ScipyOdeSimulator(self.model, range(100)).run()

    def test_conversion(self):
        m = Model.from_oldstyle(bngwiki_egfr_simple.model)
        orig = PysbFlatExporter(bngwiki_egfr_simple.model).export()
        new = PysbFlatExporter(m).export()
        for l1a, l1b in zip(orig.splitlines(), new.splitlines()):
            if l1a != l1b:
                print(f'{l1a} differs from {l1b}')
        assert orig[orig.index('\n'):] == new[new.index('\n'):]

    def test_remove_name(self):
        assert self.Model.A.name == 'A'
        assert self.Model.A._repr_no_name() == 'Monomer()'


from pysb.macros import equilibrate
import sys
class macro(object):
    def __init__(self, func):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event == 'return':
                self._locals = frame.f_locals.copy()

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals


def test_model_macro():
    class MacroModel(Model):
        A = Monomer()
        B = Monomer()
        macros = [
            (equilibrate, A(), B(), [1, 2])
        ]

    print(MacroModel)
    print(MacroModel().rules)


def test_model_merge():
    class Model1(Model):
        A = Monomer(['a'])
        B = Monomer(['b'])

    class Model2(Model):
        B = Monomer(['B'])
        C = Monomer(['c'])

    class Model3(Model1, Model2):
        pass

    print(Model3.B)
    print(Model3.export())
    print(Model3())

# TODO: Macros, model merging
