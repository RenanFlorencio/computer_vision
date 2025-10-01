# “Reconstrução” 3D com Múltiplas Imagens (T2)

**Prof. Anderson Rocha**  
**Deadline:** 30 de Setembro, 2025  

---

## Observação
É permitido (e recomendado) o uso de bibliotecas existentes (ex.: COLMAP, OpenMVG/OpenMVS, OpenCV, Open3D, MeshLab, Ceres, g2o), desde que o relatório detalhe o pipeline, parâmetros relevantes e análises.

---

## Objetivo Geral
Aplicar conceitos de Visão Computacional para reconstruir um modelo 3D (nuvem de pontos e/ou malha) de um objeto/cena a partir de múltiplas imagens, desenvolvendo competências em:
- Extração de características
- Emparelhamento
- Geometria epipolar
- Estimação de pose
- Triangulação
- (Opcional) Densificação, malha e texturização

---

## Formação dos Grupos
- Grupos de **quatro ou cinco pessoas**.

---

## Entregáveis
1. Relatório técnico completo de até **6 páginas (PDF)**.
2. Modelo 3D gerado automaticamente:
   - **Nuvem de pontos esparsa** (obrigatória).
   - Opcional: nuvem densa e/ou mesh texturizada.  
   - Demonstração em aula (vídeo de navegação de 30–60s).
3. Apresentação oral (**20–25 minutos**) com slides.
4. (Opcional) Breve vídeo explicando a técnica/pipeline (**até 10 min**).

---

## Etapas do Trabalho

### Etapa 1 — Coleta das Imagens e Preparação
- Capturar imagens com **60–80% de sobreposição**, variação angular e de altura.  
- Ambiente: luz difusa e fundo com textura.  
- Quantidade sugerida: **40–120 fotos** (pode variar conforme o objeto).  
- Fixar foco/exposição quando possível; evitar superfícies muito reflexivas ou translúcidas.  
- (Opcional) Inserir referência de escala (régua ou checkerboard).  
- No relatório: descrever ambiente, dispositivo, nº de fotos, EXIF (se relevante), estratégia de captura e desafios.

---

### Etapa 2 — Detecção e Extração de Características
- Utilizar detectores/descritores como **SIFT, ORB ou AKAZE**.  
- Visualizar **keypoints** por imagem.  
- Comparar pelo menos dois detectores/descritores e justificar a escolha final.

---

### Etapa 3 — Emparelhamento e Geometria Epipolar
- Emparelhar **features** entre pares de imagens (ex.: FLANN ou Brute Force) com **ratio test de Lowe** e, se necessário, verificação cruzada.  
- Estimar **F/E com RANSAC**; recuperar poses das câmeras (R, t) e triangular pontos 3D iniciais.  
- Exibir correspondências **inliers** e, quando conveniente, **linhas epipolares**.

---

### Etapa 4 — Reconstrução 3D e (Opcional) Densificação/Malha
- Executar **SfM** para obter nuvem esparsa e poses de câmera (ex.: COLMAP).  
- (Opcional) Executar **MVS** para densificar a nuvem de pontos.  
- (Opcional) Gerar **malha** e **texturização** (ex.: Poisson/BPA + texturing).  
- Visualizar e ajustar no **Open3D/MeshLab**: remoção de outliers, downsample, crop.

---

### Etapa 5 — Avaliação, Demonstração e Relatório
- Reportar métricas:
  - Erro de reprojeção (média/mediana).  
  - Nº de pontos esparsos/densos.  
  - Densidade (pontos/m² aprox.).  
  - Completude qualitativa.  
- Fazer **ablative study** simples: variar detector/descritor ou thresholds do RANSAC e discutir impacto.  
- Entregar:
  - Vídeo curto (**30–60s**) orbitando o modelo.  
  - Repositório reprodutível (`run.sh` chamando a pipeline + script de visualização).  

---

## Critérios de Avaliação
| Critério | Peso |
|----------|------|
| Qualidade técnica do modelo 3D | **30%** |
| Clareza e profundidade do relatório e apresentação | **20%** |
| Participação equilibrada e apresentação | **30%** |
| Criatividade ou desafios enfrentados | **20%** |

---

## Extras (Opcional)
- Recuperar escala absoluta (calibração plana de Zhang ou objeto de escala conhecido).  
- Marcar poses das câmeras no espaço 3D e exportar **GLB/OBJ** para visualização web.  
- Tratar casos difíceis: baixa textura, iluminação desafiadora, oclusões, superfícies brilhantes.  
- Comparar **patch-match** vs. métodos alternativos de densificação.  

---
