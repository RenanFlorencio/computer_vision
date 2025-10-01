import os

nomes_arquivos = os.listdir("./Trabalho 2/fotos/")
print(nomes_arquivos)
[ 'cenario3-dinofauro', 'cenario1', 'cenario5-bulbassauro5', 'cenario5-bulbassauro4',
  'cenario5-bulbassauro6-girando-no-proprio-eixo', 'cenario5-bulbassauro2', 'boneco3-compressed',
  'cenario5-bulbassauro3', 'cenario5-bulbassauro', 'boneco2', 'boneco-compressed', 'boneco', 'cenario4-vaso', 'boneco3']

import open3d as o3d
import numpy as np
import cv2 # Usaremos OpenCV para ler imagens e máscaras
import os
from PIL import Image

# ---------------- CONFIGURAÇÕES ----------------
# Substitua pelo caminho correto para seus arquivos do Colmap
COLMAP_PROJECT_PATH = "caminho/para/seu/projeto/colmap"

# Arquivos de entrada
DENSE_POINT_CLOUD_FILE = os.path.join(COLMAP_PROJECT_PATH, "dense/0/fused.ply")
SPARSE_IMAGES_FILE = os.path.join(COLMAP_PROJECT_PATH, "sparse/0/images.bin")
SPARSE_CAMERAS_FILE = os.path.join(COLMAP_PROJECT_PATH, "sparse/0/cameras.bin")
IMAGE_DIR = os.path.join(COLMAP_PROJECT_PATH, "images") # Diretório das imagens originais
MASK_DIR = os.path.join(COLMAP_PROJECT_PATH, "masks") # Diretório com as máscaras (ex: "images/img_001.png" -> "masks/img_001.png")

# Arquivo de saída
OUTPUT_POINT_CLOUD_FILE = os.path.join(COLMAP_PROJECT_PATH, "filtered_masked_cloud.ply")

# Limiar: Mínimo de imagens onde o ponto deve estar DENTRO da máscara
MIN_VIEWS = 3 
# ------------------------------------------------

def read_colmap_bin(path):
    """Lê os arquivos images.bin e cameras.bin (simplificado)."""
    # Esta é uma tarefa complexa; usaremos uma biblioteca que já faz isso se possível,
    # ou uma estrutura de dados simplificada baseada na documentação do Colmap.
    # Para simplicidade, é altamente recomendado usar o pacote pycolmap, mas aqui 
    # faremos uma leitura simplificada do images.bin para obter os dados cruciais.
    
    # É MUITO MAIS FÁCIL exportar os dados do Colmap para TEXTO (.txt) no GUI
    # e ler as matrizes de projeção diretamente. 
    raise NotImplementedError("A leitura direta de .bin é complexa. Por favor, use a exportação para .txt do Colmap e adapte o script para ler os arquivos images.txt e cameras.txt.")


def load_colmap_data_txt(sparse_path):
    """Lê cameras.txt e images.txt exportados do Colmap GUI."""
    # (Adaptado para ler .txt após exportação)
    # A estrutura de dados necessária é a matriz de projeção P = K [R | t] para cada imagem.
    
    cameras = {}
    images = {}

    # Lendo cameras.txt (Intrínsecas K)
    # Formato: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAM_1, ..., PARAM_N
    with open(os.path.join(sparse_path, "cameras.txt"), 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            camera_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            # (Simplificação: apenas para o modelo PINHOLE, que usa f, cx, cy)
            if model == 'PINHOLE':
                fx, fy, cx, cy = params
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                cameras[camera_id] = {'K': K, 'width': width, 'height': height}
            else:
                 # Adicionar suporte para outros modelos (Simple Pinhole, Radial, etc.) se necessário
                 pass


    # Lendo images.txt (Extrínsecas [R | t])
    # Formato: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME, POINTS2D[...]
    with open(os.path.join(sparse_path, "images.txt"), 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            line = lines[i]
            if line.startswith('#'): continue
            
            parts = line.strip().split()
            image_id = int(parts[0])
            Q = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            t = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
            camera_id = int(parts[8])
            image_name = parts[9]
            
            # Conversão de Quaternion (Q) para Matriz de Rotação (R)
            R = o3d.geometry.get_rotation_matrix_from_quaternion(Q[[0, 1, 2, 3]]) # O3D espera (w, x, y, z)
            
            # Matriz de Transformação da Câmera para o Mundo T_cw = [R | t]
            # No Colmap, R e t são a transformação do MUNDO para a CÂMERA (T_wc)
            # A matriz de projeção P = K * [R | t]
            
            images[image_id] = {
                'name': image_name,
                'camera_id': camera_id,
                'R': R,
                't': t.reshape((3, 1)),
                'K': cameras[camera_id]['K'],
                'width': cameras[camera_id]['width'],
                'height': cameras[camera_id]['height']
            }
            
    return images

def filter_point_cloud_by_mask():
    """Função principal para filtrar a nuvem de pontos."""
    
    # 1. Carregar a Nuvem de Pontos Densa
    print("Carregando nuvem de pontos densa...")
    try:
        pcd = o3d.io.read_point_cloud(DENSE_POINT_CLOUD_FILE)
    except Exception as e:
        print(f"ERRO ao carregar o arquivo {DENSE_POINT_CLOUD_FILE}: {e}")
        print("Certifique-se de que o arquivo existe e o Open3D consegue lê-lo (ex: PLY).")
        return
        
    points_3d = np.asarray(pcd.points)
    
    # 2. Carregar Poses e Intrínsecas do Colmap (Exportadas para .txt)
    print("Carregando dados de câmera e pose do Colmap...")
    try:
        # Você deve EXPORTAR a reconstrução esparsa para .txt no Colmap GUI:
        # File -> Export Model as Text...
        # E fornecer o caminho para a pasta que contém cameras.txt e images.txt
        colmap_data = load_colmap_data_txt(os.path.join(COLMAP_PROJECT_PATH, "sparse/0")) 
    except NotImplementedError:
        print("Por favor, exporte a reconstrução esparsa do Colmap para arquivos .txt.")
        return
    except FileNotFoundError:
        print("Certifique-se de que os arquivos 'cameras.txt' e 'images.txt' estão no diretório sparse/0.")
        return
        
    # 3. Processar Máscaras e Filtrar a Nuvem de Pontos
    
    # Matriz para contar quantas vezes um ponto cai DENTRO de uma máscara
    mask_hits = np.zeros(len(points_3d), dtype=int)
    
    total_views = len(colmap_data)
    
    for i, (image_id, data) in enumerate(colmap_data.items()):
        
        print(f"Processando imagem {i+1}/{total_views}: {data['name']}")
        
        # Carregar Máscara
        mask_path = os.path.join(MASK_DIR, data['name'])
        if not os.path.exists(mask_path):
            print(f"AVISO: Máscara não encontrada para {data['name']} em {mask_path}. Ignorando.")
            continue

        # Ler a máscara (assumindo PNG preto e branco, onde o objeto é branco - valor > 0)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        R = data['R']
        t = data['t']
        K = data['K']
        img_w = data['width']
        img_h = data['height']
        
        # 4. Projeção: Mundo (X) -> Câmera (x')
        # x' = K * [R|t] * X
        # T_wc = [R | t] (matriz 3x4)
        T_wc = np.hstack((R, t))
        
        # Adicionar coordenada homogênea (w=1) aos pontos 3D
        points_3d_hom = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        
        # Pontos no sistema de coordenadas da câmera (x_c)
        points_c = (T_wc @ points_3d_hom.T).T
        
        # Filtra pontos que estão ATRÁS da câmera (z < 0)
        valid_indices = points_c[:, 2] > 1e-6 # Z da câmera deve ser positivo
        
        points_c_valid = points_c[valid_indices]
        
        # Projeção no plano da imagem (divisão por Z)
        points_2d_proj_norm = points_c_valid[:, :2] / points_c_valid[:, 2].reshape(-1, 1)
        
        # Aplica a matriz K (intrínsecas) para obter coordenadas de pixel (u, v)
        # u = K[0,0]*x_norm + K[0,2]
        # v = K[1,1]*y_norm + K[1,2]
        # (K @ points_2d_proj_hom) - Fazer via multiplicação de matrizes é mais limpo
        
        points_2d_hom = np.hstack((points_2d_proj_norm, np.ones((len(points_2d_proj_norm), 1))))
        points_2d_pixels = (K @ points_2d_hom.T).T
        
        u = points_2d_pixels[:, 0].astype(int)
        v = points_2d_pixels[:, 1].astype(int)
        
        # 5. Filtrar pontos que caem FORA da imagem
        
        in_bounds_u = (u >= 0) & (u < img_w)
        in_bounds_v = (v >= 0) & (v < img_h)
        in_bounds = in_bounds_u & in_bounds_v
        
        u_in = u[in_bounds]
        v_in = v[in_bounds]
        
        # 6. Checagem da Máscara
        
        # Obter os índices 3D correspondentes (reaplicando os filtros)
        original_indices = np.where(valid_indices)[0][np.where(in_bounds)[0]]
        
        # Checar se a coordenada (v, u) na máscara é branca (> 0)
        # (OpenCV usa (linha, coluna) = (v, u))
        is_masked = mask[v_in, u_in] > 0
        
        # 7. Acumular contagem
        # Adicionar 1 ao contador mask_hits para todos os pontos 3D que caíram 
        # dentro dos limites da imagem E DENTRO da máscara
        
        mask_hits[original_indices[is_masked]] += 1
        
    
    # 8. Selecionar pontos finais
    
    # Seleciona todos os pontos cuja contagem de acertos na máscara é maior ou igual a MIN_VIEWS
    final_indices = mask_hits >= MIN_VIEWS
    
    # Cria a nova nuvem de pontos
    filtered_pcd = pcd.select_by_index(np.where(final_indices)[0])
    
    # Salvar
    o3d.io.write_point_cloud(OUTPUT_POINT_CLOUD_FILE, filtered_pcd)
    print(f"\nSucesso! Nuvem de pontos filtrada salva em: {OUTPUT_POINT_CLOUD_FILE}")
    print(f"Pontos originais: {len(points_3d)}, Pontos filtrados: {len(filtered_pcd.points)}")


# Para executar o script:
# filter_point_cloud_by_mask()