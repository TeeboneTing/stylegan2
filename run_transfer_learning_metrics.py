import os
import glob
import subprocess
log_path = "results/fid50k.log"


model_root = 'results/kubeflow-stylegan2-model-pvc-0e6e4fd8-263a-11ea-b9cb-000c293402fb'
model_paths = [
    '/00092-*/network*.pkl', # BP plan E
]

datasets = ['stylegan_full_BP_crop_512_tf_records ', 'stylegan_full_asuka_512_tf_records']
datasets_model_map = [0]

cmd = """python run_metrics.py
       --network {}
       --dataset {}
       --data-dir /opt/stylegan/datasets/
       --num-gpus 4"""


def execute_cmd(cmd_str):
    process_output = subprocess.run(cmd_str.split(),stdout=subprocess.PIPE)
    #dump_results(process_output.stdout,log_path)    

# def dump_results(output_str,log_path):
#     with open(log_path,"a") as f:
#         f.write(output_str + "\n")

def run():
    for idx, path in enumerate(model_paths):
        #glob.glob for last 3 pkls to save calculation time.
        pkl_paths = glob.glob(model_root + "/" + path )# [-3:]
        [print("PKL MODEL PATHS: {}".format(path_)) for path_ in pkl_paths]
        model_dir = pkl_paths[0].split("/")[-2]
        print("MODEL DIRECTORY: {}".format(model_dir))
        for pkl_file in pkl_paths:
            # Calculate fid50k
            run_cmd = cmd.format(pkl_file, datasets[datasets_model_map[idx]])
            print("="*10 + "COMMAND STRING:" + "="*10)
            print(run_cmd)
            #dump_results("MODEL DIRECTORY: {}\n".format(model_dir), log_path)
            execute_cmd(run_cmd)
            print("="*30)


if __name__ == "__main__":
    run()