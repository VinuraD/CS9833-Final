from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


class GCFExplainerRunner:
    """Utility wrapper to invoke the reference GCFExplainer implementation."""

    _DATASET_ALIASES = {
        'NCI1': 'nci1',
        'NCI109': 'nci109',
        'AIDS': 'aids',
        'PROTEINS': 'proteins',
        'MUTAGENICITY': 'mutagenicity',
        'MUTAG': 'mutagenicity',
    }

    def __init__(self,
                 dataset_name: str,
                 model_name: str,
                 output_path: str,
                 repo_root: Optional[str] = None,
                 alpha: float = 0.5,
                 theta: float = 0.05,
                 summary_theta: float = 0.1,
                 teleport: float = 0.1,
                 max_steps: int = 50000,
                 max_candidates: int = 100000,
                 device_gnn: str = '0',
                 device_neurosed: str = '0',
                 sample_size: int = 10000,
                 sample: bool = False,
                 run_summary: bool = False,
                 cf_pct: float = 0.25) -> None:
        self.original_dataset_name = dataset_name
        self.dataset_key = self._resolve_dataset_name(dataset_name)
        self.model_name = model_name
        self.alpha = alpha
        self.theta = theta
        self.summary_theta = summary_theta
        self.teleport = teleport
        self.max_steps = max_steps
        self.max_candidates = max_candidates
        self.device_gnn = device_gnn
        self.device_neurosed = device_neurosed
        self.sample_size = sample_size
        self.sample = sample
        self.run_summary = run_summary
        self.cf_pct = cf_pct
        self.repo_root = self._resolve_repo_root(repo_root)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_file = self.repo_root / 'results' / self.dataset_key / 'runs' / 'counterfactuals.pt'

    @staticmethod
    def _resolve_repo_root(repo_root: Optional[str]) -> Path:
        if repo_root:
            candidate = Path(repo_root).expanduser().resolve()
        else:
            candidate = Path(__file__).resolve().parents[2] / 'Git' / 'GCFExplainer'
        if not candidate.exists():
            raise FileNotFoundError(f'Could not locate GCFExplainer repository at {candidate}')
        return candidate

    @classmethod
    def _resolve_dataset_name(cls, dataset_name: str) -> str:
        key = dataset_name.upper()
        if key in cls._DATASET_ALIASES:
            return cls._DATASET_ALIASES[key]
        # accept already-lowercase names if compatible
        lowered = dataset_name.lower()
        if lowered in cls._DATASET_ALIASES.values():
            return lowered
        raise ValueError(f"Dataset '{dataset_name}' is not supported by GCFExplainer")

    def generate(self, force: bool = False) -> None:
        should_run = force or not self.results_file.exists()
        if should_run:
            self._run_candidates()
        else:
            print(f'Found existing GCFExplainer results at {self.results_file}, skipping regeneration.')

        if self.run_summary:
            self._run_summary()

        self._export_results()

    def _run_candidates(self) -> None:
        args = [
            sys.executable,
            'vrrw.py',
            '--dataset', self.dataset_key,
            '--alpha', str(self.alpha),
            '--theta', str(self.theta),
            '--teleport', str(self.teleport),
            '--max_steps', str(self.max_steps),
            '--k', str(self.max_candidates),
            '--device1', self.device_gnn,
            '--device2', self.device_neurosed,
            '--sample_size', str(self.sample_size),
        ]
        if self.sample:
            args.append('--sample')
        print(f'Running GCFExplainer candidate generation for dataset {self.dataset_key}...')
        self._run_subprocess(args, desc='GCFExplainer candidate generation')
        if not self.results_file.exists():
            raise FileNotFoundError(f'Expected GCFExplainer results at {self.results_file} but nothing was produced.')

    def _run_summary(self) -> None:
        args = [
            sys.executable,
            'summary.py',
            '--dataset', self.dataset_key,
            '--theta', str(self.summary_theta),
            '--device', self.device_gnn,
        ]
        print(f'Running GCFExplainer summary evaluation for dataset {self.dataset_key}...')
        self._run_subprocess(args, desc='GCFExplainer summary generation')

    def _run_subprocess(self, args, desc: str) -> None:
        try:
            subprocess.run(args, cwd=self.repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'{desc} failed with exit code {exc.returncode}') from exc

    def _export_results(self) -> None:
        if not self.results_file.exists():
            raise FileNotFoundError(
                f'GCFExplainer results {self.results_file} were not found; ensure candidate generation completed successfully.'
            )
        shutil.copy2(self.results_file, self.output_path)
        metadata = {
            'source_results': str(self.results_file.resolve()),
            'dataset': self.original_dataset_name,
            'dataset_key': self.dataset_key,
            'model_name': self.model_name,
            'cf_pct': self.cf_pct,
            'alpha': self.alpha,
            'theta': self.theta,
            'summary_theta': self.summary_theta,
            'teleport': self.teleport,
            'max_steps': self.max_steps,
            'max_candidates': self.max_candidates,
            'device_gnn': self.device_gnn,
            'device_neurosed': self.device_neurosed,
            'sample_size': self.sample_size,
            'sample_mode': self.sample,
            'run_summary': self.run_summary,
        }
        metadata_path = self.output_path.with_suffix(self.output_path.suffix + '.meta.json')
        with open(metadata_path, 'w', encoding='utf-8') as meta_file:
            json.dump(metadata, meta_file, indent=2)
        print(f'Copied GCFExplainer results to {self.output_path} (metadata: {metadata_path}).')
