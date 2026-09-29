"""Headless run of the stand-alone Mode builder page, plus the mode-building core."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from sevendt_calc import config, modebuild


def test_build_mode_validation_and_json():
    filtinfo = config.load_filtinfo()
    seq = {'7DT01': ['g', 'm400'], '7DT02': ['g', 'm425'], '7DT03': ['m386', 'g']}
    built = modebuild.build_mode('Spec', 'my mode!', seq, filtinfo)
    assert built.name == 'my_mode' and built.file_name == 'my_mode.specmode'
    assert not built.errors and any('Units without a filter' in w for w in built.warnings)
    assert built.obsmode.units == ['7DT01', '7DT02', '7DT03'] and built.obsmode.filter_multiplicity()['g'] == 3
    text = built.to_json()
    assert text.startswith('{\n    "7DT01": ["g", "m400"],\n') and '"7DT16": []' in text
    import json
    assert json.loads(text)['7DT03'] == ['m386', 'g']
    assert json.loads(built.to_json(include_empty=False)).keys() == {'7DT01', '7DT02', '7DT03'}
    bad = modebuild.build_mode('Color', 'c', {'7DT01': ['m425'], '7DT02': ['g', None, '']}, filtinfo)
    assert bad.errors and 'm425 not installed' in bad.errors[0] and bad.filters_by_unit['7DT02'] == ['g']
    assert bad.file_name == 'c.colormode'
    rows = modebuild.filter_summary(built)
    assert [r['filter'] for r in rows][0] == 'g' and rows[0]['slots'] == 3


def test_add_remove_and_pad():
    seq = modebuild.add_filter({}, '7DT01', 'g')
    seq = modebuild.add_filter(seq, '7DT01', 'g')
    seq = modebuild.add_filter(seq, '7DT01', 'm400')
    assert seq == {'7DT01': ['g', 'g', 'm400']}              # duplicates are kept, in click order
    assert modebuild.remove_last(seq, '7DT01') == {'7DT01': ['g', 'g']}
    assert modebuild.remove_last({}, '7DT02') == {}
    assert modebuild.remove_at(seq, '7DT01', 0) == {'7DT01': ['g', 'm400']}
    assert modebuild.remove_at(seq, '7DT01', 5) == seq and modebuild.remove_at(seq, '7DT09', 0) == seq
    padded = modebuild.pad_sequences({'7DT01': ['g', 'm400', 'm650'], '7DT02': ['g'], '7DT03': []})
    assert padded == {'7DT01': ['g', 'm400', 'm650'], '7DT02': ['g', 'g', 'g'], '7DT03': []}


def test_parse_mode_file():
    kind, name, seq = modebuild.parse_mode_file(b'{"7DT01": ["g", "g"], "7DT02": []}', 'GAMA test.colormode')
    assert (kind, name) == ('Color', 'GAMA_test') and seq == {'7DT01': ['g', 'g'], '7DT02': []}
    assert modebuild.parse_mode_file('{"7DT01": ["m650"]}', 'specnew.specmode')[0] == 'Spec'
    for bad in (b'not json', b'[1, 2]', b'{"7DT01": "g"}'):
        try:
            modebuild.parse_mode_file(bad, 'x.specmode')
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_interactive_mode_figure():
    from sevendt_calc import modeplot
    om = config.resolve_obsmode('Spec', 'specall')
    fig = modeplot.plot_obsmode_interactive(om)
    assert len(fig.layout.shapes) == sum(len(f) for f in om.filters_by_unit.values())


def test_modebuilder_page_renders():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'modebuilder_app.py'), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == '7DT Mode builder'
    assert any('No filter selected yet' in w.value for w in at.warning)
    at.button(key='mb_load').click()          # 'Empty' template: still no filters
    at.run()
    assert not at.exception, at.exception
    assert at.session_state['mb_kind'] == 'Spec' and at.session_state['mb_name'] == 'newmode'
    at.selectbox(key='mb_template_Spec').set_value('specall')
    at.button(key='mb_load').click()
    at.run()
    assert not at.exception, at.exception
    assert any('newmode.specmode' in m.value for m in at.markdown)
    assert not at.error
    assert at.session_state['mb_seq']['7DT01'] == ['m650', 'm769w']
    def click(row, unit):                         # a cell selection in the grid table
        at.session_state['mb_grid'] = {'selection': {'rows': [], 'columns': [], 'cells': [[row, unit]]}}
        at.run()
        assert not at.exception, at.exception
    click(0, '7DT01')                             # slot 1 of 7DT01 = g -> appended
    assert at.session_state['mb_seq']['7DT01'] == ['m650', 'm769w', 'g']
    click(0, '7DT01')                             # again: g twice
    assert at.session_state['mb_seq']['7DT01'] == ['m650', 'm769w', 'g', 'g']
    assert '"7DT01": ["m650", "m769w", "g", "g"]' in at.code[0].value
    click(5, '7DT01')                             # slot 6 of 7DT01 is empty: nothing happens
    assert at.session_state['mb_seq']['7DT01'] == ['m650', 'm769w', 'g', 'g']
    click(9, '7DT01')                             # Order 1 cell of 7DT01 (m650): removed
    assert at.session_state['mb_seq']['7DT01'] == ['m769w', 'g', 'g']
    assert at.session_state['mb_grid']['selection']['cells'][0][1] in ('7DT01', '7DT02')   # parked on the spacer row
    click(11, '7DT01')                            # Order 3 cell (the last g): removed
    assert at.session_state['mb_seq']['7DT01'] == ['m769w', 'g']
    click(13, '7DT01')                            # the spacer row: ignored
    assert at.session_state['mb_seq']['7DT01'] == ['m769w', 'g']
    assert not any('different numbers of filters' in w.value for w in at.warning)   # all units have 2 again
    click(0, '7DT01')                             # a third filter on 7DT01 only
    assert any('different numbers of filters' in w.value for w in at.warning)
    at.checkbox(key='mb_pad').check()
    at.run()
    assert not any('different numbers' in w.value for w in at.warning)
    at.button(key='mb_clear').click()
    at.run()
    assert at.session_state['mb_seq'] == {}
