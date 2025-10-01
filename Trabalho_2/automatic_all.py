import subprocess
import os

# PREFIX = "seg_"
# PREFIX = ""
for prefix in ["", "seg_"]:
    PREFIX = prefix
    
    BASE_PATH = f"./Trabalho_2/{PREFIX}fotos/"
    if not os.path.exists(BASE_PATH):
        BASE_PATH = f"./{PREFIX}fotos/"
    
    nomes_cenas = sorted(os.listdir(BASE_PATH))
    print(nomes_cenas)
    k = 0
    for nome_cena in nomes_cenas:
        CENARIO = nome_cena  
        print(f"Processing scenario: {CENARIO} with prefix '{PREFIX}' ({k+1}/{len(nomes_cenas)})")
        IMAGES_PATH = BASE_PATH + CENARIO + "/"
        MASKS_PATH = "./mask_fotos/" + CENARIO + "/"
        DATABASE_PATH = f"./{PREFIX}processed/{CENARIO}/database.db"
        OUTPUT_PATH = f"./{PREFIX}processed/{CENARIO}"
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        env_clean = os.environ.copy()
        if 'LD_LIBRARY_PATH' in env_clean:
            del env_clean['LD_LIBRARY_PATH']

        #feature extraction
        comando = f"colmap feature_extractor --database_path {DATABASE_PATH} --image_path {IMAGES_PATH} --ImageReader.single_camera 1 \
            --SiftExtraction.max_image_size 3400 "
        subprocess.run(comando, shell=True, env=env_clean)

        # Emparelhamento de características
        comando = f"colmap exhaustive_matcher --database_path {DATABASE_PATH}"
        subprocess.run(comando, shell=True, env=env_clean)

        # Mapper
        comando = f"mkdir -p {OUTPUT_PATH}/sparse"
        subprocess.run(comando, shell=True, env=env_clean)

        comando = f"colmap mapper --database_path {DATABASE_PATH} --image_path {IMAGES_PATH} --output_path {OUTPUT_PATH}/sparse/ --Mapper.num_threads 8"
        subprocess.run(comando, shell=True, env=env_clean)

        # Isso vai salvar os pontos em um txt na pasta sparse/0
        comando = f"colmap model_converter --input_path {OUTPUT_PATH}/sparse/0 --output_path {OUTPUT_PATH}/sparse/0 --output_type TXT"
        subprocess.run(comando, shell=True, env=env_clean)

        # Se o modelo tiver falhado, tente converter os outros modelos gerados
        for i in range(1, 5):
            if os.path.exists(f'{OUTPUT_PATH}/sparse/{i}'):
                comando = f"colmap model_converter --input_path {OUTPUT_PATH}/sparse/{i} --output_path {OUTPUT_PATH}/sparse/{i} --output_type TXT"
                subprocess.run(comando, shell=True, env=env_clean)
            else:
                break

        # preprocessing (gera undistorted images)
        for i in range(1,5):
            if not os.path.exists(f'{OUTPUT_PATH}/sparse/{i}'):
                break
            comando = f"colmap image_undistorter --image_path {IMAGES_PATH} --input_path {OUTPUT_PATH}/sparse/{i} \
            --output_path {OUTPUT_PATH}/dense/ --output_type COLMAP"
            subprocess.run(comando, shell=True, env=env_clean)

        comando = f"colmap image_undistorter --image_path {IMAGES_PATH} --input_path {OUTPUT_PATH}/sparse/0 \
            --output_path {OUTPUT_PATH}/dense/ --output_type COLMAP"
        subprocess.run(comando, shell=True, env=env_clean)

        # PatchMatch stereo
        comando = f"colmap patch_match_stereo --workspace_path {OUTPUT_PATH}/dense/ --workspace_format COLMAP --PatchMatchStereo.geom_consistency true"
        subprocess.run(comando, shell=True, env=env_clean)

        # Fusion dos depth maps
        comando = f'colmap stereo_fusion --workspace_path {OUTPUT_PATH}/dense/ --output_path {OUTPUT_PATH}/dense/fused.ply'
        subprocess.run(comando, shell=True, env=env_clean)
        
        k += 1