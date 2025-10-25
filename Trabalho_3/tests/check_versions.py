"""
check_versions.py - Verificar versões de todas as dependências
Projeto: Sistema de Navegação Assistida
"""

import sys

def check_version(package_name, import_name=None):
    """Verificar versão de um pacote"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'N/A')
        print(f"✅ {package_name:20s} {version}")
        return version
    except ImportError:
        print(f"❌ {package_name:20s} NÃO INSTALADO")
        return None
    except Exception as e:
        print(f"⚠️  {package_name:20s} ERRO: {e}")
        return None

def main():
    """Verificar todas as dependências do projeto"""
    print("="*70)
    print("VERIFICAÇÃO DE VERSÕES - Sistema de Navegação Assistida")
    print("="*70)
    
    print(f"\n🐍 Python: {sys.version.split()[0]}")
    
    print("\n--- Deep Learning ---")
    torch_version = check_version("torch")
    torchvision_version = check_version("torchvision")
    
    print("\n--- Computer Vision ---")
    check_version("opencv-python", "cv2")
    check_version("numpy")
    check_version("pillow", "PIL")
    
    print("\n--- YOLO / Models ---")
    check_version("ultralytics")
    check_version("timm")
    
    print("\n--- Text-to-Speech ---")
    check_version("pyttsx3")
    
    print("\n--- Utilitários ---")
    check_version("matplotlib")
    check_version("tqdm")
    
    # Verificar CUDA
    print("\n--- GPU / CUDA ---")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA disponível: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Count: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA não disponível - usando CPU")
    except Exception as e:
        print(f"⚠️  Não foi possível verificar CUDA: {e}")
    
    # Verificar compatibilidade PyTorch + torchvision
    print("\n--- Compatibilidade ---")
    if torch_version and torchvision_version:
        # Versões esperadas
        expected_pairs = [
            ("2.5.1", "0.20.1"),
            ("2.5.0", "0.20.0"),
            ("2.4.1", "0.19.1"),
        ]
        
        compatible = False
        for torch_v, tv_v in expected_pairs:
            if torch_v in torch_version and tv_v in torchvision_version:
                compatible = True
                break
        
        if compatible:
            print("✅ PyTorch e torchvision são compatíveis")
        else:
            print("⚠️  Verifique compatibilidade PyTorch/torchvision")
            print("   https://github.com/pytorch/vision#installation")
    
    print("\n" + "="*70)
    print("Verificação concluída!")
    print("="*70)

if __name__ == "__main__":
    main()
