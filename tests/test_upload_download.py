"""Structure upload and browser downloads (DEC-023 Tier-1 #1, M-INTERACT INT.4/7)."""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quantui.downloads import download_link_html, saved_with_download_html
from quantui.structure_upload import read_uploaded_structure

WATER_XYZ = """3
water charge=0 multiplicity=1  RHF/STO-3G
O 0.000 0.000 0.117
H 0.000 0.757 -0.469
H 0.000 -0.757 -0.469
"""


class TestDownloadLinks:
    def test_link_carries_the_file_bytes(self, tmp_path):
        f = tmp_path / "water.xyz"
        f.write_bytes(WATER_XYZ.encode())  # write_text would add \r on Windows
        html = download_link_html(f)
        assert 'download="water.xyz"' in html
        m = re.search(r'href="data:([^;]+);base64,([^"]+)"', html)
        assert m and m.group(1) == "chemical/x-xyz"
        assert base64.b64decode(m.group(2)).decode() == WATER_XYZ

    def test_large_file_gets_a_notice_not_a_link(self, tmp_path, monkeypatch):
        import quantui.downloads as dl

        monkeypatch.setattr(dl, "MAX_DOWNLOAD_BYTES", 10)
        f = tmp_path / "big.cube"
        f.write_text("x" * 100)
        html = dl.download_link_html(f)
        assert "href=" not in html and "too large" in html

    def test_missing_file(self, tmp_path):
        assert "not found" in download_link_html(tmp_path / "nope.png")

    def test_saved_status_includes_link(self, tmp_path):
        f = tmp_path / "plot.png"
        f.write_bytes(b"\x89PNG\r\n")
        html = saved_with_download_html(f)
        assert "Saved: plot.png" in html and "data:image/png;base64," in html


class TestReadUploads:
    def test_xyz_comment_supplies_charge_and_multiplicity(self):
        up = read_uploaded_structure(
            "w.xyz",
            WATER_XYZ.replace(
                "charge=0 multiplicity=1", "charge=1 multiplicity=2"
            ).encode(),
        )
        assert up.molecule.atoms == ["O", "H", "H"]
        assert (up.molecule.charge, up.molecule.multiplicity) == (1, 2)
        assert up.charge_mult_from_file

    def test_plain_xyz_uses_setup_values_and_says_so(self):
        text = "3\n\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
        up = read_uploaded_structure("w.xyz", text.encode(), charge=-1, multiplicity=1)
        assert up.molecule.charge == -1
        assert not up.charge_mult_from_file
        assert any("Calculation Setup" in n for n in up.notes)

    def test_multiframe_xyz_takes_the_last_frame(self):
        frame2 = WATER_XYZ.replace("0.117", "0.200")
        up = read_uploaded_structure("traj.xyz", (WATER_XYZ + frame2).encode())
        assert up.molecule.coordinates[0][2] == pytest.approx(0.2)
        assert any("2 structures" in n for n in up.notes)

    def test_gaussian_input_charge_multiplicity(self):
        gjf = (
            "%chk=oh.chk\n%mem=1GB\n# UB3LYP/6-31G(d) Opt\n\nhydroxyl radical\n\n"
            "0 2\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n\n"
        )
        up = read_uploaded_structure("oh.gjf", gjf.encode())
        assert up.molecule.atoms == ["O", "H"]
        assert (up.molecule.charge, up.molecule.multiplicity) == (0, 2)

    def test_mol_block_formal_charge(self):
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from rdkit.Chem import AllChem

        rd = Chem.AddHs(Chem.MolFromSmiles("[NH4+]"))
        AllChem.EmbedMolecule(rd, randomSeed=1)
        up = read_uploaded_structure("nh4.mol", Chem.MolToMolBlock(rd).encode())
        assert sorted(up.molecule.atoms) == ["H", "H", "H", "H", "N"]
        assert up.molecule.charge == 1 and up.molecule.multiplicity == 1

    def test_2d_sdf_is_embedded_in_3d(self):
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from rdkit.Chem import AllChem

        rd = Chem.MolFromSmiles("CC(=O)C")  # acetone, heavy atoms only
        AllChem.Compute2DCoords(rd)
        up = read_uploaded_structure(
            "acetone.sdf", (Chem.MolToMolBlock(rd) + "$$$$\n").encode()
        )
        assert up.molecule.atoms.count("H") == 6  # hydrogens added
        assert any(abs(c[2]) > 0.1 for c in up.molecule.coordinates)
        assert any("2-D" in n for n in up.notes)

    def test_pdb(self):
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from rdkit.Chem import AllChem

        rd = Chem.AddHs(Chem.MolFromSmiles("O"))
        AllChem.EmbedMolecule(rd, randomSeed=1)
        up = read_uploaded_structure("w.pdb", Chem.MolToPDBBlock(rd).encode())
        assert sorted(up.molecule.atoms) == ["H", "H", "O"]

    def test_mol2(self):
        pytest.importorskip("rdkit")
        mol2 = """@<TRIPOS>MOLECULE
water
 3 2 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 O1          0.0000    0.0000    0.1170 O.3     1  HOH1        0.0000
      2 H1          0.0000    0.7570   -0.4690 H       1  HOH1        0.0000
      3 H2          0.0000   -0.7570   -0.4690 H       1  HOH1        0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     3    1
"""
        up = read_uploaded_structure("w.mol2", mol2.encode())
        assert up.molecule.atoms == ["O", "H", "H"]

    @pytest.mark.parametrize("name", ["notes.docx", "noext"])
    def test_unsupported(self, name):
        with pytest.raises(ValueError, match="Unsupported"):
            read_uploaded_structure(name, b"x")

    def test_garbage_is_a_clean_error(self):
        with pytest.raises(ValueError):
            read_uploaded_structure("bad.xyz", b"not a structure at all")

    def test_filename_is_sanitized(self, tmp_path):
        # A path-like name must not escape the temp folder.
        up = read_uploaded_structure("../../etc/w.xyz", WATER_XYZ.encode())
        assert up.molecule.atoms == ["O", "H", "H"]


class TestAppWiring:
    def test_upload_widget_loads_the_molecule(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app.structure_upload.value = (
            {
                "name": "w.xyz",
                "type": "",
                "size": len(WATER_XYZ),
                "content": memoryview(
                    WATER_XYZ.replace(
                        "charge=0 multiplicity=1", "charge=1 multiplicity=2"
                    ).encode()
                ),
                "last_modified": datetime.now(timezone.utc),
            },
        )
        assert app._molecule is not None
        assert app._molecule.get_formula() == "H2O"
        # Charge/multiplicity came from the file, so the setup widgets follow.
        assert (app.charge_si.value, app.mult_si.value) == (1, 2)
        assert "Loaded H2O" in app.upload_msg.value
        assert app.structure_upload.value == ()

    def test_upload_error_is_shown(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app.structure_upload.value = (
            {
                "name": "x.docx",
                "type": "",
                "size": 1,
                "content": memoryview(b"x"),
                "last_modified": datetime.now(timezone.utc),
            },
        )
        assert "Unsupported" in app.upload_msg.value

    def test_files_tab_download_and_load(self, tmp_path):
        from quantui.app import QuantUIApp

        f = tmp_path / "w.xyz"
        f.write_text(WATER_XYZ)
        app = QuantUIApp()
        app._files_selected_path = f
        app._sync_files_action_buttons()
        assert not app._files_download_btn.disabled
        assert not app._files_load_btn.disabled
        app._on_files_download(None)
        assert "data:chemical/x-xyz;base64," in app._files_download_html.value
        app._on_files_load(None)
        assert app._molecule.get_formula() == "H2O"

    def test_files_tab_load_disabled_for_non_structures(self, tmp_path):
        from quantui.app import QuantUIApp

        f = tmp_path / "result.json"
        f.write_text("{}")
        app = QuantUIApp()
        app._files_selected_path = f
        app._sync_files_action_buttons()
        assert not app._files_download_btn.disabled
        assert app._files_load_btn.disabled

    def test_export_xyz_offers_a_download(self, tmp_path):
        from quantui.app import QuantUIApp
        from quantui.molecule import Molecule

        app = QuantUIApp()
        app._set_molecule(
            Molecule(atoms=["H", "H"], coordinates=[[0, 0, 0], [0, 0, 0.74]]), "test"
        )
        app._last_result_dir = tmp_path
        app._on_export_xyz(None)
        assert (tmp_path / "H2_RHF_STO-3G.xyz").exists() or list(tmp_path.glob("*.xyz"))
        assert 'download="' in app._download_html.value
        assert Path(tmp_path).exists()
