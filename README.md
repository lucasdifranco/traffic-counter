# Contagem de veículos por sentido em câmera de trânsito

Este trabalho tem como objetivo contar os veículos que cruzam uma linha virtual em uma
transmissão de câmera de trânsito, separando-os por sentido e registrando cada cruzamento.
O pipeline combina o detector YOLO26m, o tracker TrackTrack com re-identificação e uma
contagem por linha definida em arquivo de configuração, e as decisões, os trade-offs e as
limitações estão documentados no `SOLUTION.md`.

## Como rodar

Requer Python 3.12 ou superior. O detector e o PyTorch são instalados à parte, e o peso do
detector é baixado na primeira execução caso esteja ausente.

```bash
pip install -r requirements.txt
pip install ultralytics
# GPU (opcional, mas recomendado):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

```bash
python test_crossing.py                          # testes da lógica de cruzamento
python pick_line.py                              # marca a linha e grava line.json
python stream_process.py                          # transmissão ao vivo
python stream_process.py -u samples/clip.mp4      # arquivo local
```

Na janela, a barra de espaço pausa e retoma a execução, e as teclas `q` ou `Esc` a
encerram. A saída consiste no `events.jsonl`, com um evento por linha, e no `summary.json`,
com os totais por classe e por sentido.

## Arquivos

| caminho | conteúdo |
|---|---|
| `stream_process.py` | pipeline: detecção, rastreamento, contagem e overlay |
| `crossing.py` | lógica de cruzamento da linha, sem dependência de OpenCV |
| `test_crossing.py` | testes da lógica de cruzamento |
| `pick_line.py` | ferramenta para marcar a linha e gerar o `line.json` |
| `bench_stages.py` | medição do tempo de processamento por etapa |
| `trackers/traffic.yaml` | configuração do TrackTrack adotada |
| `line.json` | linha virtual marcada sobre a cena |
| `samples/evidence_30s.mp4` | clipe de 32 s com o resultado do pipeline |
| `samples/flowchart.png` | fluxo do pipeline |
| `SOLUTION.md` | decisões, trade-offs, limitações e números |

## Resultados

A contagem foi avaliada contra uma contagem manual de 60 s sobre o `samples/clip.mp4`, com
1.800 quadros. Os totais por sentido coincidiram com a referência manual, restando um erro
de rótulo de classe em um veículo. A acurácia detalhada e o tempo de processamento por
etapa estão no `SOLUTION.md`.
