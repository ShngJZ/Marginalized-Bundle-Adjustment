import os, natsort

def txt2seqs(seqs_path):
    seqs = open(seqs_path).readlines()
    seqs = [x.rstrip('\n') for x in seqs]
    seqs = [f"seq-{x[8::].zfill(2)}" for x in seqs]
    return seqs

def read_split_txt(seqs_path):
    seqs = open(seqs_path).readlines()
    seqs = [x.rstrip('\n') for x in seqs]
    seqs = [f"seq-{x[8::].zfill(2)}" for x in seqs]
    return seqs

def read_scene_query_seqs(data_root):
    sceens = ['stairs', 'fire', 'office', 'pumpkin', 'heads', 'redkitchen', 'chess']
    scenes_path = [os.path.join(data_root, x) for x in sceens]
    seqs_qrys = list()
    for scene_path in scenes_path:
        qry_txt = os.path.join(scene_path, 'TestSplit.txt')
        assert os.path.exists(qry_txt)
        seqs_qry = read_split_txt(qry_txt)
        for x in seqs_qry:
            seqs_qrys.append([os.path.basename(scene_path), x])
    seqs_qrys = natsort.natsorted(seqs_qrys)
    return seqs_qrys

def acquire_marker_query_map(idx2fname_mapper, seqs_map, seqs_qry):
    """
    Generate a marker map indicating whether each entry is a map or query sequence

    Args:
        idx2fname_mapper: List of index-to-filename mappings
        seqs_map: List of map sequence names
        seqs_qry: List of query sequence names

    Returns:
        List of markers ('map' or 'qry') corresponding to input mapper

    Raises:
        ValueError: If sequence name is not found in either map or query lists
    """
    marker_query_map = list()

    for x in idx2fname_mapper:
        # Extract sequence name from mapper entry
        seq_name = x.split(' ')[1].split('/')[0]

        # Classify as map or query sequence
        if seq_name in seqs_map:
            marker_query_map.append('map')
        elif seq_name in seqs_qry:
            marker_query_map.append('qry')
        else:
            raise ValueError(f"{seq_name} not found in map and query")

    return marker_query_map