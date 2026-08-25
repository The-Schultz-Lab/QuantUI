"""Tests for interactive XYZ input wiring."""

from quantui.app import QuantUIApp
from quantui.app_xyz_input import on_load_xyz, on_xyz_cleanup, on_xyz_cleanup_accept


def _make_app():
    return QuantUIApp()


class TestAppXyzInput:
    def test_xyz_widgets_exist(self):
        app = _make_app()
        for name in (
            "xyz_add_atom_btn",
            "xyz_apply_table_btn",
            "xyz_cleanup_btn",
            "xyz_fill_table_btn",
            "xyz_table_box",
            "xyz_cleanup_preview_box",
        ):
            assert hasattr(app, name)

    def test_load_single_nitrogen_succeeds(self):
        app = _make_app()
        app.xyz_area.value = "N 0 0 0"
        app.charge_si.value = 0
        app.mult_si.value = 1
        on_load_xyz(app)
        assert app._molecule is not None
        assert app._molecule.get_formula() == "N"
        assert "multiplicity" in app.xyz_msg.value.lower()

    def test_load_preserves_charge_mult_fields(self):
        app = _make_app()
        app.charge_si.value = -1
        app.mult_si.value = 2
        app.xyz_area.value = "N 0 0 0"
        on_load_xyz(app)
        assert app.charge_si.value == -1
        assert app.mult_si.value == 2
        assert app._molecule.charge == -1
        assert app._molecule.multiplicity == 2

    def test_cleanup_accept_updates_textarea(self):
        app = _make_app()
        app.xyz_area.value = "n 0 0 0"
        on_xyz_cleanup(app)
        assert app.xyz_cleanup_preview_box.layout.display != "none"
        on_xyz_cleanup_accept(app)
        assert app.xyz_area.value.strip().startswith("N")
        assert app.xyz_cleanup_preview_box.layout.display == "none"
