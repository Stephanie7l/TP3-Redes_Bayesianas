# Arquivos para análise

Todas as saídas são recalculadas por `python tp3.py`. As probabilidades nos CSVs usam escala de **0 a 1**, exceto campos explicitamente chamados `percentual` e `diferenca_pp`. Os CSVs usam vírgula como separador e ponto decimal; no Excel, use **Dados → De Texto/CSV** se a abertura direta não separar as colunas.

## Imagens principais — `resultados/`

Cada figura existe em PNG (300 dpi), SVG e PDF vetorial.

| Nome, sem extensão | Uso |
|---|---|
| `rede_manual` | Hipótese de relações definida antes do aprendizado |
| `rede_hc` | Rede aprendida por Hill-Climbing/BIC |
| `rede_chow_liu` | Árvore aprendida por Chow–Liu |
| `sensibilidade` | Mudança na chance de G3 baixa ao retirar uma evidência do caso |

## Figuras complementares — `resultados/figuras/`

| Nome, sem extensão | Uso |
|---|---|
| `distribuicao_g3` | Frequências relativas das faixas finais no treino e no teste |
| `consultas_g3` | Comparação das distribuições posteriores de cada cenário |
| `matriz_confusao_hc` | Contagens de faixas reais e previstas no teste, HC |
| `matriz_confusao_chow_liu` | Contagens de faixas reais e previstas no teste, Chow–Liu |
| `avaliacao_teste` | Acurácia e log-loss dos métodos e baseline |
| `cenarios_alternativos_g2` | Distribuições para estados alternativos de G2 no caso explicado |
| `cobertor_markov_g3` | Destaque de G3 e dos nós que compõem seu cobertor |

## Tabelas — `resultados/tabelas/`

| CSV | Conteúdo |
|---|---|
| `limites_discretizacao` | Quantis estimados no treino e regra de cada faixa |
| `frequencias_categorias` | Contagem e percentual de cada estado, variável e conjunto |
| `consultas_com_evidencias` | Cenários, evidências em JSON, suporte no treino e posteriores |
| `comparacao_estruturas` | BIC, quantidade de arestas e parâmetros livres |
| `arestas_aprendidas` | Lista de origem e destino das arestas por método |
| `comparacao_com_manual` | Arestas concordantes, invertidas, ausentes e novas |
| `cpts_hc` e `cpts_chow_liu` | CPTs completas em formato longo: estado, configuração dos pais e probabilidade |
| `predicoes_teste` | Índice original, classe real, classe prevista e probabilidades de cada método |
| `matriz_confusao_hc` e `matriz_confusao_chow_liu` | Matriz numérica, linhas reais e colunas previstas |
| `metricas_teste` | Acurácia e log-loss, incluindo baseline |
| `cenarios_alternativos_g2` | Distribuições condicionais alternativas do caso |
| `caso_explicado` | Evidências, faixa real e probabilidades do registro escolhido |

Na comparação com a rede manual, `invertidas` registra a orientação originalmente proposta. `ausentes` e `novas` comparam pares sem orientação, em ordem alfabética. Essas contagens avaliam concordância com uma hipótese, não acurácia contra uma estrutura verdadeira conhecida.

## Outros arquivos — `resultados/`

- `resultados.json`: resumo completo, incluindo escolhas, limites, grafos, caso e ambiente.
- `treino_discretizado.csv` e `teste_discretizado.csv`: conjuntos usados, com índice original a partir de zero.
- `comparacao.csv`, `consultas.csv`, `sensibilidade.csv`: tabelas compactas da execução principal.
- `cpts_hc.json` e `cpts_chow_liu.json`: valores completos e ordem dos estados de cada CPT.
- `cpts_hc.txt` e `cpts_chow_liu.txt`: visualização textual. Tabelas largas podem ser abreviadas pelo pgmpy; use os JSONs ou os CSVs completos para conferir todos os valores.

## Relação com as tarefas

1. Dados e preparação: limites, frequências, conjuntos discretizados e fonte da base.
2. Estrutura manual: imagem `rede_manual` e constante `ARESTAS_MANUAIS` em `tp3.py`.
3. Estruturas aprendidas: redes, comparação de BIC, listas de arestas e comparação com a proposta.
4. Parâmetros e inferência: CPTs e consultas com suporte das evidências.
5. Explicação: caso selecionado, sensibilidade, cobertor e cenários condicionais alternativos.
