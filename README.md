# Trabalho Prático 3 — Redes Bayesianas

Universidade Federal de Minas Gerais · Escola de Engenharia  
EEE048 — Fundamentos de Inteligência Artificial  
Curso de Engenharia de Sistemas

## Integrantes

- Cecília Rodrigues Saraiva
- Caio César Araújo Guimarães
- Stéphanie Pereira Barbosa
- Thales Eduardo Dias de Souza

## Objetivo

Analisar a distribuição da nota final de Português usando a base Student Performance (UCI). O código percorre as cinco tarefas do TP: preparação dos dados, proposta manual de rede, aprendizado com Hill-Climbing/BIC e Chow–Liu, estimação de CPTs e inferência, e explicação de um caso por cobertor de Markov e sensibilidade.

O pacote contém código e arquivos de análise. O relatório será preparado pelo grupo a partir desses resultados.

## Executar no Windows

1. Instale **Python 3.13 de 64 bits**, caso ainda não tenha essa versão.
2. Extraia **todo o ZIP** para uma pasta do computador.
3. Abra `instalar.bat` e aguarde a instalação. Esta etapa usa internet.
4. Abra `executar.bat` para gerar os resultados. A pasta será aberta ao final.

Nas próximas execuções, basta abrir `executar.bat`. Os arquivos de análise são atualizados na mesma pasta. A base original está incluída e a análise funciona sem internet depois de instalar as dependências.

Os arquivos `.bat` foram revisados, mas a execução verificada neste ambiente foi feita em Python no Linux; a confirmação no Windows deve ser feita no computador do grupo.

### Alternativa pelo CMD

Abra o CMD na pasta extraída:

```bat
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tp3.py
```

No Linux/macOS, use `python3.13 -m venv .venv` e `.venv/bin/python` no lugar de `.venv\Scripts\python.exe`.

## Organização

| Arquivo ou pasta | Conteúdo |
|---|---|
| `tp3.py` | Preparação, hipótese manual, aprendizado, CPTs, inferências e explicação |
| `analises.py` | Exportação das tabelas, matrizes e figuras complementares |
| `TP3_Redes_Bayesianas.ipynb` | Execução guiada, usando as funções do código principal |
| `instalar.bat` | Cria o ambiente e instala as dependências no Windows |
| `executar.bat` | Executa a análise e abre a pasta de resultados |
| `requirements.txt` | Dependências da análise com versões fixadas |
| `requirements-notebook.txt` | Dependências adicionais para abrir o notebook |
| `dados/` | CSV original, dicionário e fonte da base |
| `resultados/` | Saídas já geradas e atualizadas a cada execução |
| `ARQUIVOS_GERADOS.md` | Mapa das saídas para ajudar na análise do grupo |

## Notebook (opcional)

O script e o `.bat` fazem a análise completa, sem precisar abrir o notebook. Para usar a versão guiada:

```bat
.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.venv\Scripts\python.exe -m jupyter lab
```

Abra `TP3_Redes_Bayesianas.ipynb` na pasta do projeto e execute as células em ordem. Ele importa `tp3.py` e `analises.py`, evitando manter duas cópias diferentes da implementação.

## Resultados de referência

| Medida | Hill-Climbing | Chow–Liu |
|---|---:|---:|
| BIC de treino | -4118,1152 | -4127,2485 |
| Arestas | 8 | 9 |
| Parâmetros livres | 46 | 54 |
| Acurácia no teste | 77,69% | 77,69% |
| Log-loss no teste | 0,4917 | 0,4917 |

Foram usados 649 registros, com 519 no treino e 130 no teste. A semente é 42. O modelo selecionado pelo BIC do treino foi Hill-Climbing. Os dois métodos acertaram 101 registros no teste. O baseline marginal teve 43,85% de acurácia.

Os arquivos de resultados já estão incluídos para consulta e serão recalculados ao executar o programa. As figuras são exportadas em **PNG de 300 dpi, SVG e PDF vetorial**. Esses PDFs são figuras individuais, não um relatório.

## Leitura do código

Comece por `tp3.py`: as funções seguem a sequência das tarefas. `preparar_dados` faz o tratamento, `rede` monta o grafo, `desenhar` cria sua imagem, `comparar` confere as arestas, `consultar` faz uma inferência e `executar` organiza a análise. O arquivo `analises.py` reúne apenas as exportações complementares.

O Hill-Climbing é chamado diretamente do pgmpy. As contas e percursos principais usam laços explícitos para facilitar a leitura. O ajuste de CPTs para nós isolados foi mantido porque é necessário para lidar com o comportamento da versão usada do pgmpy.

A semente da divisão dos dados é 42. `executar.bat` também fixa `PYTHONHASHSEED=42`. Em empates de score, outra configuração de processo ou versão da biblioteca pode escolher uma orientação equivalente diferente. Por isso, use as figuras e os números produzidos na mesma execução ao analisar o resultado.

## Decisões de modelagem

- Dez variáveis: age, Medu, studytime, failures, schoolsup, higher, absences, G1, G2 e G3.
- Cortes nos quantis 1/3 e 2/3 do treino para idade, faltas e notas. Os limites não são recalculados no teste.
- Empates ficam na mesma faixa. Por isso os grupos podem ter tamanhos diferentes; em faltas, a faixa baixa contém os zeros.
- Categorias naturais de escolaridade e estudo são preservadas; reprovações são agrupadas em nenhuma ou uma ou mais.
- Hill-Climbing inicia sem arestas, com BIC, até dois pais e no máximo 10.000 iterações. É usada diretamente a implementação da biblioteca, sem classes personalizadas.
- Chow–Liu aprende uma árvore por informação mútua, orientada a partir de G3. A raiz não implica causalidade.
- CPTs com BDeu, amostra equivalente 5, e inferência exata por eliminação de variáveis.
- Caso explicado: primeiro registro do teste, por índice original, com G2 baixa.

## Cuidados para a análise

“G3 baixa” significa nota final até 11 pelos cortes usados, e não necessariamente reprovação. Usar G2 corresponde a acompanhamento após o segundo período. Setas aprendidas descrevem dependências estatísticas e não comprovam causalidade. Os cenários que trocam G2 são consultas condicionais, não intervenções. A sensibilidade é local, não aditiva e não mede importância global.

O enunciado pede aprovação prévia da base pelos professores; a escolha ainda precisa ser confirmada pelo grupo. A origem dos dados está em `dados/FONTE.md`. O grupo deve revisar as interpretações e observar as orientações da disciplina sobre assistência de IA.

## Subir para o GitHub pelo CMD

Crie um repositório vazio no GitHub, sem adicionar README ou `.gitignore` pelo site. Depois, no CMD aberto nesta pasta:

```bat
git init
git add .
git status
git commit -m "Adiciona TP3 de redes bayesianas"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

Substitua a URL pela do seu repositório. Confira o `git status` antes do commit. O `.gitignore` exclui o ambiente virtual e arquivos temporários; código, notebook, dados e resultados são versionados. Se usar uma pasta que já pertence a um repositório, siga a configuração existente, sem repetir `git init` e `git remote add`.

## Verificação

O código valida aciclicidade, compatibilidade das CPTs, normalização das probabilidades, suporte das consultas e equivalência da inferência com todas as evidências versus apenas o cobertor de Markov. As predições individuais e matrizes de confusão permitem conferir as métricas. O fluxo completo e as nove células do notebook foram executados com Python 3.13.5 no Linux antes do empacotamento. As dependências foram também verificadas para Windows de 64 bits com Python 3.13. Os arquivos .bat precisam ser executados no Windows do grupo.

## Referências

- Cortez, P. (2008). Student Performance [Dataset]. UCI. https://doi.org/10.24432/C5TG7T
- Hill-Climbing: https://pgmpy.org/_modules/pgmpy/estimators/HillClimbSearch.html
- Chow–Liu: https://pgmpy.org/_modules/pgmpy/estimators/TreeSearch.html
- Parâmetros: https://pgmpy.org/api/parameter_estimation.html
