import sympy
import copy
from pysb.core import Model as OldModel, \
    Component, \
    Monomer as OldMonomer, \
    MonomerPattern as OldMonomerPattern, \
    Parameter as OldParameter, \
    Compartment as OldCompartment, \
    Rule as OldRule, \
    Initial as OldInitial, \
    Expression as OldExpression, \
    Observable as OldObservable, \
    Tag as OldTag, \
    MultiState, \
    extract_site_conditions
from pysb.annotation import Annotation
import collections
import re
import networkx as nx


def _remove_name(s):
    return re.sub(r'\'\w+\'(, )?', '', s, 1)


class Monomer(OldMonomer):
    def __init__(self, *args, **kwargs):
        super(Monomer, self).__init__(None, *args, _export=False)

    def _repr_no_name(self):
        return _remove_name(super(Monomer, self).__repr__())


class MonomerPattern(OldMonomerPattern):
    def is_concrete(self):
        # Cannot verify compartments until Model is instantiated
        return self.is_site_concrete()

    def __call__(self, conditions=None, **kwargs):
        """Build a new MonomerPattern with updated site conditions. Can be used
        to obtain a shallow copy by passing an empty argument list."""
        # The new object will have references to the original monomer and
        # compartment, and a shallow copy of site_conditions which has been
        # updated according to our args (as in Monomer.__call__).
        site_conditions = self.site_conditions.copy()
        site_conditions.update(extract_site_conditions(conditions, **kwargs))
        return MonomerPattern(self.monomer, site_conditions, self.compartment)


class Parameter(OldParameter):
    def __new__(cls, *args, **kwargs):
        return super(sympy.Symbol, cls).__new__(cls)

    def __init__(self, *args):
        if len(args) == 1:
            args = (None, args[0], False)
        super(Parameter, self).__init__(*args)

    def _repr_no_name(self):
        return _remove_name(super(Parameter, self).__repr__())


class Compartment(OldCompartment):
    def __init__(self, *args, **kwargs):
        super(Compartment, self).__init__(None, *args, **kwargs, _export=False)

    def _repr_no_name(self):
        return _remove_name(super(Compartment, self).__repr__())


class Rule(OldRule):
    def __init__(self, *args, **kwargs):
        super(Rule, self).__init__(None, *args, **kwargs, _export=False)

    def _repr_no_name(self):
        return _remove_name(super(Rule, self).__repr__())


class Expression(OldExpression):
    def __new__(cls, *args, **kwargs):
        return super(sympy.Symbol, cls).__new__(cls)

    def __init__(self, *args):
        if len(args) == 1:
            args = (None, args[0], False)
        super(Expression, self).__init__(*args)

    def _repr_no_name(self):
        return _remove_name(super(Expression, self).__repr__())


class Observable(OldObservable):
    def __new__(cls, *args, **kwargs):
        return super(sympy.Symbol, cls).__new__(cls)

    def __init__(self, *args):
        if len(args) == 1:
            args = (args[0], 'molecules')
        if len(args) == 2:
            args = (None, args[0], args[1], False)
        super(Observable, self).__init__(*args)

    def _repr_no_name(self):
        return _remove_name(super(Observable, self).__repr__())


class Initial(OldInitial):
    def __init__(self, *args, **kwargs):
        super(Initial, self).__init__(*args, **kwargs, _export=False)


def _yield_elements_by_type(components, component_type):
    for c in components:
        if isinstance(c, component_type):
            yield c


def sort_components(components):
    g = nx.DiGraph()
    g.add_nodes_from(components)
    for c in components:
        for d in c.component_dependencies():
            g.add_edge(d, c)

    from networkx.algorithms.dag import topological_sort
    return topological_sort(g)


class ModelDict(collections.OrderedDict):
    def __setitem__(self, key, value):
        if isinstance(value, Component):
            # print(f'Setting {value}.name to {key}')
            value.name = key
        elif key == 'macros':
            for macro in value:
                if not callable(macro[0]):
                    raise ValueError('Macro not callable')
                for component in macro[0](*macro[1:]):
                    super(ModelDict, self).__setitem__(component.name,
                                                       component)

        super(ModelDict, self).__setitem__(key, value)


class ModelMeta(type):
    @classmethod
    def __prepare__(mcs, name, bases):
        return ModelDict()


class Model(OldModel, metaclass=ModelMeta):
    initials = []
    annotations = []
    macros = []

    @staticmethod
    def _convert_components_old_to_new(obj_dict):
        obj_mapping = {}
        new_obj_dict = {}
        for c_name, c_orig in obj_dict.items():
            if isinstance(c_orig, OldMonomer):
                c = Monomer(copy.copy(c_orig.sites),
                            copy.deepcopy(c_orig.site_states))
            elif isinstance(c_orig, OldParameter):
                c = Parameter(c_orig.value)
            elif isinstance(c_orig, OldInitial):
                pat = c_orig.pattern()
                for mp in pat.monomer_patterns:
                    mp.monomer = obj_mapping[mp.monomer]
                c = Initial(pat, obj_mapping[c_orig.value], c_orig.fixed)
            elif isinstance(c_orig, OldExpression):
                c = Expression(c_orig.expr.xreplace(new_obj_dict))
            elif isinstance(c_orig, OldCompartment):
                c = Compartment(obj_mapping.get(c_orig.parent, None),
                                c_orig.dimension,
                                obj_mapping.get(c_orig.size, None))
            elif isinstance(c_orig, OldObservable):
                rp = copy.copy(c_orig.reaction_pattern)
                rp.complex_patterns = [cp.copy() for cp in rp.complex_patterns]
                for cp in rp.complex_patterns:
                    for mp in cp.monomer_patterns:
                        mp.monomer = obj_mapping[mp.monomer]
                c = Observable(rp, c_orig.match)
            elif isinstance(c_orig, OldRule):
                rexp = copy.copy(c_orig.rule_expression)
                rexp.reactant_pattern = copy.copy(rexp.reactant_pattern)
                rexp.product_pattern = copy.copy(rexp.product_pattern)
                for p in (rexp.reactant_pattern, rexp.product_pattern):
                    p.complex_patterns = [cp.copy() for cp in p.complex_patterns]
                    for cp in p.complex_patterns:
                        for mp in cp.monomer_patterns:
                            mp.monomer = obj_mapping[mp.monomer]
                c = Rule(rexp,
                         obj_mapping[c_orig.rate_forward],
                         obj_mapping.get(c_orig.rate_reverse, None),
                         c_orig.delete_molecules, c_orig.move_connected)
            else:
                raise ValueError('Unknown component type: {}'.format(
                    type(c_orig)))

            obj_mapping[c_orig] = c
            new_obj_dict[c_name] = c

        return new_obj_dict, obj_mapping

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs, _export=False)

        obj_props = {k: getattr(self, k) for k in dir(self)}
        # Filter to components
        c_dict = {c: name for name, c in obj_props.items()
                  if isinstance(c, Component)}
        # Get sort order
        c_sorted = sort_components(c_dict.keys())

        # print(self.equilibrate_A_to_B.component_dependencies())

        # Get obj_dict

        for k, v in c_dict.items():
            print(k, v)
        #print(c_sorted)
        obj_dict = collections.OrderedDict(
            (c_dict[c], c) for c in c_sorted
        )

        # Use __dict__ instead of dir() to preserve addition order
        # obj_dict = collections.OrderedDict(
        #     (k, getattr(self, k)) for k in self.__class__.__dict__.()
        #     if isinstance(v, Component)
        # )

        name_mapping, obj_mapping = self._convert_components_old_to_new(obj_dict)
        for c_name, c_new in name_mapping.items():
            c_new.name = c_name
            c_new.model = self
            self.add_component(c_new)

        print(obj_mapping)

        # Initials
        for i in self._copy_initials(self.__class__.initials, obj_mapping):
            self.add_initial(i)

        # Annotations
        obj_mapping[self.__class__] = self  # Mapping for model itself
        for a in self._copy_annotations(self.__class__.annotations, obj_mapping):
            self.add_annotation(a)

        # TODO: Check compartments specified if model has compartments

    @staticmethod
    def _copy_initials(initials_list, obj_mapping):
        """" Copy initials and update species targets """
        # print('Original initials list:')
        # print(initials_list)
        new_initials = []
        for initial in initials_list:
            cp = initial.pattern
            for mp in cp.monomer_patterns:
                mp.monomer = obj_mapping[mp.monomer]
            # print(f'Mapping {initial.value} to {obj_mapping[initial.value]}')
            new_initials.append(
                Initial(cp, obj_mapping[initial.value], initial.fixed))
        # print('New initials list;')
        # print(new_initials)
        return new_initials

    @staticmethod
    def _copy_annotations(annotations_list, obj_mapping):
        new_annotations = []
        for annotation in annotations_list:
            new_annotations.append(
                Annotation(
                    obj_mapping[annotation.subject],
                    copy.copy(annotation.object),
                    annotation.predicate,
                    _export=False
                )
            )
        return new_annotations

    @classmethod
    def from_oldstyle(cls, model):
        name_mapping, obj_mapping = cls._convert_components_old_to_new(
            {c.name: c for c in model.components}
        )
        kls = type('MyModel', (Model, ), name_mapping)

        # Since we didn't use __setitem__, set names manually
        for c_name in name_mapping:
            getattr(kls, c_name).name = c_name

        kls.initials = cls._copy_initials(model.initials, obj_mapping)

        # Annotations can target the model - need to add that to mapping
        obj_mapping[model] = kls
        kls.annotations = cls._copy_annotations(model.annotations, obj_mapping)

        print(kls.export())

        return kls()

    @classmethod
    def export(cls):
        s = 'class {}(Model):\n'.format(cls.__name__)
        for c_name, c in cls.__dict__.items():
            if isinstance(c, Component):
                print(str(c))
                s += '    {} = {}\n'.format(c_name, c._repr_no_name())

        s += '    initials = [\n'
        for initial in cls.initials:
            s += '        {},\n'.format(initial)
        s += '    ]\n'

        s += '    annotations = [\n'
        for annotation in cls.annotations:
            s += '        {},\n'.format(annotation)
        s += '    ]\n'

        return s