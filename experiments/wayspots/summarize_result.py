import json
import os, argparse
import numpy as np
from tabulate import tabulate
from collections import OrderedDict

def format_output(aggregated_results):
    keys = ["scene", '10cm/5deg', '50cm/5deg', '5deg', '5cm']
    aggregated_results_sorted = list()
    for x in aggregated_results:
        x_ = OrderedDict()
        for k in keys:
            x_[k] = x[k]
        aggregated_results_sorted.append(x_)
    return aggregated_results_sorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')
    parser.add_argument(
        '--data-root', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/wayspots"
    )
    parser.add_argument(
        '--output-location', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/wayspots"
    )
    parser.add_argument(
        '--depth-model', type=str, choices=['ZoeDepth', 'UniDepth', 'DUSt3R'], default='DUSt3R'
    )
    parser.add_argument(
        '--corres-model', type=str, choices=['RoMa', 'MASt3R'], default='RoMa'
    )

    args = parser.parse_args()
    scenes = [
        'wayspots_bears',
        'wayspots_cubes',
        'wayspots_inscription',
        'wayspots_lawn',
        'wayspots_map',
        'wayspots_squarebench',
        'wayspots_statue',
        'wayspots_tendrils',
        'wayspots_therock',
        'wayspots_wintersign'
    ]
    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )
    aggregated_results = OrderedDict()
    for scene in scenes:
        sfm_perscene = os.path.join(f"{preprocess_location}_sfm", scene)
        evals = os.path.join(sfm_perscene, "evals.json")
        if not os.path.exists(evals):
            continue
        evals = json.load(open(evals, "r"))
        for idx, result in enumerate(evals):
            result.update({'scene': scene})
            result_key = f"{result['loss_key']}_{result['iteration']}"
            if result_key not in aggregated_results:
                aggregated_results[result_key] = list()
            aggregated_results[result_key].append(result)

    for key in aggregated_results:
        aggregated_result = aggregated_results[key]
        average = {'scene': 'average'}
        for key in ['10cm/5deg', '50cm/5deg', '5deg', '5cm']:
            average[key] = np.mean(np.array([x[key] for x in aggregated_result]))
        aggregated_result.append(average)

    for key in aggregated_results:
        assert aggregated_results[key][-1]['scene'] == 'average'

    err_rot_tls = [(aggregated_results[x][-1]['10cm/5deg'] + aggregated_results[x][-1]['10cm/5deg']) for x in aggregated_results]
    keys = [x for x in aggregated_results]

    maxid = np.argmax(np.array(err_rot_tls))
    maxkey = keys[maxid]

    print(tabulate(format_output(aggregated_results[maxkey]), headers="keys"))