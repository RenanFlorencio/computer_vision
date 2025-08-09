# Panoramas

* **Professor:** Anderson Rocha
* **Prazo Final:** 31 de Agosto de 2025
* **Grupo:** [@gabijacob](https://github.com/gabijacob), [@RaphaelSVSouza](https://github.com/RaphaelSVSouza), [@RenanFlorencio](https://github.com/RenanFlorencio), [@yobinad1](https://github.com/yobinad1) e [@IgorEBatista](https://github.com/IgorEBatista)

## Objetivo Geral

O objetivo deste trabalho é aplicar conceitos de Visão Computacional para criar panoramas a partir de múltiplas imagens. O projeto visa desenvolver competências práticas em detecção de características, emparelhamento, estimação de homografia e *blending*.

## Entregáveis

1.  Um relatório técnico completo de até seis páginas em PDF.
2.  Panoramas gerados automaticamente que deverão ser apresentados em aula.
3.  Uma apresentação oral de 20 a 25 minutos com slides e um vídeo de 10 minutos demonstrando a técnica.

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

## Critérios de Avaliação

| Critério | Peso |
| :--- | :--- |
| Qualidade técnica do panorama | 30% |
| Clareza e profundidade do relatório e apresentação | 20% |
| Participação equilibrada e apresentação | 30% |
| Criatividade ou desafios enfrentados | 20% |

## Extras (opcional)

* Remoção de fantasmas (*ghosting*).
* Detecção automática da ordem correta das imagens.
* Geração de panorama $360^{\circ}$ (caso as imagens permitam).