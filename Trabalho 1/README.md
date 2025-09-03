# Panoramas

* **Professor:** Anderson Rocha
* **Prazo Final:** 31 de Agosto de 2025
* **Grupo:** [@gabijacob](https://github.com/gabijacob), [@RaphaelSVSouza](https://github.com/RaphaelSVSouza), [@RenanFlorencio](https://github.com/RenanFlorencio), [@yobinad1](https://github.com/yobinad1) e [@IgorEBatista](https://github.com/IgorEBatista)

## Objetivo Geral

O objetivo deste trabalho é aplicar conceitos de Visão Computacional para criar panoramas a partir de múltiplas imagens. O projeto visa desenvolver competências práticas em detecção de características, emparelhamento, estimação de homografia e *blending*.

## Etapas do Trabalho

### Etapa 1: Coleta das Imagens 
* Capturar um conjunto mínimo de cinco imagens parcialmente sobrepostas usando uma câmera ou celular, em um mesmo ambiente (interno ou externo).
* Garantir boas condições de iluminação e qualidade.
* Incluir no relatório o local, o dispositivo utilizado e as observações da coleta.

### Etapa 2: Detecção e Extração de Características
* Utilizar detectores como SIFT, ORB ou AKAZE para extrair pontos de interesse.
* Visualizar as características em cada imagem.
* Comparar pelo menos dois detectores e justificar a escolha.
* Mostrar os *keypoints* detectados sobrepostos nas imagens.

### Etapa 3: Emparelhamento de Características
* Aplicar algoritmos de emparelhamento, como FLANN ou Brute Force Matcher.
* Utilizar filtros como o *ratio test* de David Lowe.
* Exibir imagens com linhas conectando os *keypoints* correspondentes.
* Relatar os desafios encontrados, como *keypoints* errados ou objetos em movimento.

### Etapa 4: Estimação de Homografia e Alinhamento
* Calcular a homografia usando RANSAC.
* Alinhar as imagens com `warpPerspective` (OpenCV) ou outro método.
* Incluir visualizações intermediárias que mostrem o alinhamento progressivo.

### Etapa 5: Composição e *Blending*
* Implementar a composição do panorama.
* Aplicar *blending* para suavizar as transições (por exemplo: *feathering*, linear, *multiband*).
* Avaliar visualmente e com métricas simples, como a continuidade de linhas e a ausência de distorções.

### Resultado Final

Emparelhamento e Alinhmaento das imagens:

<img src=output/morfologia/panorama_morfologia.jpg />

Composição e *Blending*:

<img src=output/final/panorama_final.jpg />

### Report

For further details, check ``report.pdf``.
