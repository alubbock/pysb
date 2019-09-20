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
    raise

# TODO: Macros, model merging
