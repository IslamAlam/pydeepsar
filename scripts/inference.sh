#!/bin/bash
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --gres=gpu:1

# Export the environment variable
export TF_CPP_MIN_LOG_LEVEL=3

source ~/.conda/bin/activate pydeepsar


delay=5
counter=0

# input="main-train-20230130-*"
basefolder=${1:-"/dss/dsshome1/01/di93sif/di93sif/Summit_East_Coast_2017/TanDEM-X/"}
folderpattern=${2:-"TDM1_SAR__COS_BIST_SM_S_SRA_*"}
inference_model_path=${3:-"/dss/dsshome1/01/di93sif/di93sif/ice/pydeepsar/models/model-20240427-165512.185"}
inference_model_path=${3:-"/dss/dsshome1/01/di93sif/di93sif/ice/pydeepsar/models/model-weibull-20240502-173505.203"}
# inference_model_path=${3:-"/dss/dsshome1/01/di93sif/di93sif/ice/pydeepsar/models/model-weibull-20240502-194400.391"}
model_folder=$(basename $(dirname "${inference_model_path}"))
model_folder=$(basename "${inference_model_path}")

layer_names=${4:-"d_pen,coherence,PhaseCenterDepth,phase"}
layer_names=${4:-"lambda_w,k_w,coherence,PhaseCenterDepth,phase"}


echo find "${basefolder}" -name "${folderpattern}" -type d
for i in $(find "${basefolder}" -name "${folderpattern}" -type d); do
	echo "$i/"

	srun \
		--job-name="Inference Plots" --cluster=hpda2 --partition=hpda2_compute_gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --mem=150G --time=04:00:00  --label  \
    python -m pydeepsar.models.inference --inference_model_path "${inference_model_path}" --tandem_root "$i" --layer_names "${layer_names}"  --reference_bias "bias" --y_reference "iodem3_2017_DEM" --y_estimated "SEC_and_cnst_offset_postp_dem" 2>> "$i/${model_folder}_errorOutput.log" >> "$i/${model_folder}_output.log" &

		# python model_inference.py --weights="$i/" --scenes_path='/dss/dsshome1/01/di93sif/di93sif/GEDI_TDX/lope_reproc_20221108.v01' --lvis_path='/dss/dsshome1/01/di93sif/di93sif/LVIS/Afrisar_LVIS_Biomass_VProfiles_1775/data' --lvis_site='lope' --lvis_rh=98 --filter_scenes='*20160125T173041*' 2>> "$i/errorOutput.log" >> "$i/output.log" &

	let counter++
	if (( counter % 6 == 0 )); then
    sleep 30
    # sleep 1800
	fi

	sleep ${delay}

done

wait

# python -m pydeepsar.models.inference --inference_model_path "/dss/dsshome1/01/di93sif/di93sif/ice/pydeepsar/models/model-20240427-165512.185" --tandem_root "/dss/dsshome1/01/di93sif/di93sif/Summit_East_Coast_2017" --layer_names "d_pen,coherence,PhaseCenterDepth,phase" --reference_bias "bias" --y_reference "iodem3_2017_DEM" --y_estimated "SEC_and_cnst_offset_postp_dem"
