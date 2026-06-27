import unittest
import os
import json
import shutil
from dvc import DatasetVCS


class TestDatasetVersionControl(unittest.TestCase):
    """
    Suite de pruebas unitarias para validar las funciones de control de versiones
    de datasets (commits, snapshots, checkout, diffs).
    """

    def setUp(self) -> None:
        self.store_dir = ".dvc_store_test"
        self.work_file = "test_dataset.json"
        
        # Inicializamos el VCS en carpeta de pruebas
        self.vcs = DatasetVCS(store_dir=self.store_dir)
        self.vcs.init()
        
        # Creamos un dataset inicial para pruebas
        self.initial_data = [
            {"id": 1, "text": "fact first", "value": 10},
            {"id": 2, "text": "fact second", "value": 20},
            {"id": 3, "text": "fact third", "value": 30}
        ]
        
        with open(self.work_file, "w", encoding="utf-8") as f:
            json.dump(self.initial_data, f)

    def tearDown(self) -> None:
        # Limpieza de archivos temporales
        if os.path.exists(self.work_file):
            os.remove(self.work_file)
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_init_repository(self) -> None:
        """
        Verifica que se creen las carpetas y archivos base del repositorio de datos.
        """
        self.assertTrue(os.path.exists(self.store_dir))
        self.assertTrue(os.path.exists(os.path.join(self.store_dir, "objects")))
        self.assertTrue(os.path.exists(self.vcs.metadata_file))
        self.assertTrue(os.path.exists(self.vcs.head_file))

    def test_commit_and_checkout(self) -> None:
        """
        Prueba el flujo de commit de datos y restauracion mediante checkout.
        """
        # Hacemos el primer commit
        commit1 = self.vcs.commit(self.work_file, "Commit inicial con 3 registros")
        self.assertTrue(len(commit1) > 0)
        
        # Modificamos el dataset agregando un registro y guardamos
        modified_data = list(self.initial_data)
        modified_data.append({"id": 4, "text": "fact fourth", "value": 40})
        with open(self.work_file, "w", encoding="utf-8") as f:
            json.dump(modified_data, f)
            
        # Hacemos el segundo commit
        commit2 = self.vcs.commit(self.work_file, "Commit 2 con 4 registros")
        self.assertNotEqual(commit1, commit2)
        
        # Comprobamos que el archivo de trabajo tiene 4 registros actualmente
        with open(self.work_file, "r", encoding="utf-8") as f:
            current_data = json.load(f)
        self.assertEqual(len(current_data), 4)
        
        # Hacemos checkout al commit 1
        self.vcs.checkout(commit1, self.work_file)
        
        # Comprobamos que el archivo de trabajo se restauro a los 3 registros originales
        with open(self.work_file, "r", encoding="utf-8") as f:
            restored_data = json.load(f)
        self.assertEqual(len(restored_data), 3)
        self.assertEqual(restored_data[0]["id"], 1)

    def test_diff_calculation(self) -> None:
        """
        Prueba la comparacion semantica de diferencias entre dos commits de datos.
        """
        # Commit A (Original)
        commit_a = self.vcs.commit(self.work_file, "Commit A")
        
        # Commit B (Modificado):
        # - Agrega ID 4
        # - Elimina ID 2
        # - Modifica ID 3
        new_data = [
            {"id": 1, "text": "fact first", "value": 10},
            {"id": 3, "text": "fact third MODIFIED", "value": 35},
            {"id": 4, "text": "fact fourth", "value": 40}
        ]
        with open(self.work_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f)
            
        commit_b = self.vcs.commit(self.work_file, "Commit B")
        
        # Calculamos diferencias
        diff_result = self.vcs.diff(commit_a, commit_b)
        
        self.assertEqual(diff_result["type"], "json_records")
        self.assertEqual(diff_result["added_records_count"], 1)
        self.assertEqual(diff_result["removed_records_count"], 1)
        self.assertEqual(diff_result["modified_records_count"], 1)
        self.assertEqual(diff_result["total_records_before"], 3)
        self.assertEqual(diff_result["total_records_after"], 3)


if __name__ == "__main__":
    unittest.main()
