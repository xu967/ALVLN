#!/usr/bin/env python3
"""Regression: filtered RGB step gap preserves source IDs and excludes fake sequences."""
import json
import tempfile
from pathlib import Path

from test_dataset_windows import _make_episode
from dataset import EpisodeSequenceDataset


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        _make_episode(root)
        ep=root/'910001'
        path=ep/'actions.jsonl'
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        rows[2]['rgb_ok']=False  # valid frames: 0,1,3,4,5
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        ds=EpisodeSequenceDataset(root,seq_len=3,image_size=8,require_depth=False)
        steps=[x['step'] for x in ds.samples]
        assert steps == [0,1,5],steps  # 3-frame window [3,4,5]
        assert ds.samples[-1]['label']==0
        assert ds.samples[-1]['rgb_paths'][-1].name=='rgb_00005.png'
        print('step-gap regression: PASS',steps)

if __name__=='__main__':main()
