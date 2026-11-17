"""TP3 - redes bayesianas aplicadas ao desempenho escolar.

Execute: python tp3.py. Os resultados são gravados na pasta resultados.
"""
from pathlib import Path
import json
import platform
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from pgmpy.estimators import HillClimbSearch, BIC, TreeSearch, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

BASE = Path(__file__).resolve().parent
SEMENTE = 42
COLUNAS = ['age', 'Medu', 'studytime', 'failures', 'schoolsup', 'higher',
           'absences', 'G1', 'G2', 'G3']
NUMERICAS = ['age', 'absences', 'G1', 'G2', 'G3']
ROTULOS = ['baixa', 'media', 'alta']
# Hipóteses definidas antes de rodar os algoritmos. Não são conclusões causais.
ARESTAS_MANUAIS = [
    ('Medu', 'higher'), ('higher', 'studytime'), ('studytime', 'G1'),
    ('failures', 'G1'), ('schoolsup', 'G1'), ('absences', 'G3'),
    ('G1', 'G2'), ('G2', 'G3'), ('age', 'failures')]


def preparar_dados():
    bruto = pd.read_csv(BASE / 'dados/student-por.csv', sep=';')
    dados = bruto[COLUNAS].copy()
    treino, teste = train_test_split(dados, test_size=0.2, random_state=SEMENTE)
    treino, teste = treino.copy(), teste.copy()
    faltantes = dados.isna().sum().to_dict()
    preenchimento, cortes = {}, {}
    # Ajustamos cada transformação só no treino, sem usar o conjunto de teste.
    for coluna in COLUNAS:
        if coluna in NUMERICAS:
            valor = treino[coluna].median()
        else:
            valor = treino[coluna].mode().iloc[0]
        preenchimento[coluna] = str(valor)
        treino[coluna] = treino[coluna].fillna(valor)
        teste[coluna] = teste[coluna].fillna(valor)
    for coluna in NUMERICAS:
        limites = treino[coluna].quantile([0, 1/3, 2/3, 1]).to_numpy()
        if limites[1] >= limites[2]:
            raise ValueError(f'Quantis internos repetidos em {coluna}; rever discretização.')
        cortes[coluna] = limites.tolist()
        # Os extremos abertos recebem também valores futuros fora da faixa do treino.
        limites[0], limites[-1] = -np.inf, np.inf
        for tabela in (treino, teste):
            tabela[coluna] = pd.cut(tabela[coluna], bins=limites,
                                    labels=ROTULOS, include_lowest=True).astype(str)
    for tabela in (treino, teste):
        # Reprovações são contagens concentradas em zero: usamos categorias naturais.
        tabela['failures'] = np.where(tabela['failures'].astype(int) == 0, 'nenhuma', 'uma_ou_mais')
        for coluna in COLUNAS:
            tabela[coluna] = tabela[coluna].astype(str)
    estados = {}
    for coluna in COLUNAS:
        if coluna in NUMERICAS:
            estados[coluna] = ROTULOS.copy()
        else:
            estados[coluna] = sorted(dados[coluna].dropna().astype(str).unique())
    estados['failures'] = ['nenhuma', 'uma_ou_mais']
    resumo = {'n_total': len(dados), 'n_treino': len(treino), 'n_teste': len(teste),
              'faltantes': faltantes, 'duplicatas_selecionadas': int(dados.duplicated().sum()),
              'duplicatas_completas': int(bruto.duplicated().sum()),
              'cortes': cortes, 'preenchimento': preenchimento, 'estados': estados}
    return treino, teste, resumo


def rede(arestas):
    modelo = DiscreteBayesianNetwork()
    modelo.add_nodes_from(COLUNAS)  # Nós sem arestas também fazem parte do modelo.
    modelo.add_edges_from(arestas)
    return modelo


def desenhar(modelo, titulo, destino):
    fig, ax = plt.subplots(figsize=(8, 5))
    grafo_desenho = nx.DiGraph()
    grafo_desenho.add_nodes_from(modelo.nodes())
    grafo_desenho.add_edges_from(modelo.edges())
    pos = nx.spring_layout(grafo_desenho, seed=SEMENTE, k=1.2)
    assert all(np.isfinite(xy).all() for xy in pos.values())
    for no, (x, y) in pos.items():
        ax.text(x, y, no, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.6', fc='#d9edf4', ec='#456575'))
    for u, v in modelo.edges():
        ax.annotate('', xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle='->', color='#456575',
                                    shrinkA=25, shrinkB=25, lw=1.4))
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(titulo, fontsize=14, pad=15)
    ax.axis('off')
    fig.tight_layout()
    from analises import salvar_figura
    salvar_figura(fig, destino)


def comparar(manual, aprendido):
    # O esqueleto evita contar orientação invertida como ausência de associação.
    esq_m = {frozenset(a) for a in manual.edges()}
    esq_a = {frozenset(a) for a in aprendido.edges()}
    return {'mesma_direcao': sorted(set(manual.edges()) & set(aprendido.edges())),
            'invertidas': sorted((u, v) for u, v in manual.edges() if aprendido.has_edge(v, u)),
            'ausentes': sorted(tuple(sorted(a)) for a in esq_m - esq_a),
            'novas': sorted(tuple(sorted(a)) for a in esq_a - esq_m),
            'jaccard_esqueleto': len(esq_m & esq_a) / len(esq_m | esq_a)}


def consultar(inferencia, evidencia):
    dist = inferencia.query(['G3'], evidence=evidencia, show_progress=False)
    return {s: float(dist.get_value(G3=s)) for s in ROTULOS}


def executar():
    saida = BASE / 'resultados'
    saida.mkdir(exist_ok=True)
    treino, teste, resumo = preparar_dados()
    treino.to_csv(saida / 'treino_discretizado.csv', index=True, index_label='indice_original')
    teste.to_csv(saida / 'teste_discretizado.csv', index=True, index_label='indice_original')
    manual = rede(ARESTAS_MANUAIS)
    desenhar(manual, 'Estrutura proposta antes do aprendizado', saida / 'rede_manual.png')
    # BIC maior é melhor. No máximo dois pais mantêm as CPTs pequenas.
    hc = HillClimbSearch(treino).estimate(scoring_method=BIC(treino),
            max_indegree=2, max_iter=10000, show_progress=False)
    cl = TreeSearch(treino, root_node='G3', n_jobs=1).estimate(
            estimator_type='chow-liu', show_progress=False)
    modelos = {'Hill-Climbing': rede(hc.edges()), 'Chow-Liu': rede(cl.edges())}
    score = BIC(treino)
    comparacoes = []
    for nome, modelo in modelos.items():
        assert nx.is_directed_acyclic_graph(modelo)
        modelo.fit(treino, estimator=BayesianEstimator, prior_type='BDeu',
                   equivalent_sample_size=5, state_names=resumo['estados'], n_jobs=1)
        # No pgmpy 1.0.0, o estimador pode ignorar nós isolados.
        # Neles, a CPT é a marginal com o mesmo prior BDeu (ESS=5).
        for no in modelo.nodes():
            if modelo.get_cpds(no) is None:
                estados_no = resumo['estados'][no]
                contagens = treino[no].value_counts().reindex(estados_no, fill_value=0).to_numpy()
                probabilidades = (contagens + 5 / len(estados_no)) / (len(treino) + 5)
                modelo.add_cpds(TabularCPD(no, len(estados_no), probabilidades[:, None],
                                          state_names={no: estados_no}))
        assert modelo.check_model()
        for cpd in modelo.get_cpds():
            assert np.allclose(cpd.get_values().sum(axis=0), 1)
        parametros_livres = 0
        for cpd in modelo.get_cpds():
            configuracoes_pais = int(np.prod(cpd.cardinality[1:]))
            parametros_livres += (cpd.variable_card - 1) * configuracoes_pais
        comparacoes.append({'metodo': nome, 'bic_treino': float(score.score(modelo)),
                            'arestas': len(modelo.edges()),
                            'parametros_livres': int(parametros_livres)})
        desenhar(modelo, nome, saida / ('rede_hc.png' if nome == 'Hill-Climbing' else 'rede_chow_liu.png'))
    # Escolhemos pelo BIC do treino; o teste fica reservado para avaliação.
    escolhido = max(comparacoes, key=lambda x: x['bic_treino'])['metodo']
    modelo = modelos[escolhido]
    inf = VariableElimination(modelo)
    cenarios = [({}, 'Sem evidência'), ({'G2': 'baixa'}, 'G2 baixa'),
                ({'G2': 'alta'}, 'G2 alta'),
                ({'G2': 'baixa', 'failures': 'uma_ou_mais'}, 'G2 baixa e reprovações'),
                ({'higher': 'yes', 'studytime': '3'}, 'Deseja graduação e estuda 5 a 10h'),
                ({'absences': 'alta'}, 'Faltas altas')]
    consultas = []
    for ev, nome in cenarios:
        if ev:
            assert (treino[list(ev)] == pd.Series(ev)).all(axis=1).any()
        prob = consultar(inf, ev)
        assert np.isclose(sum(prob.values()), 1)
        suporte = int((treino[list(ev)] == pd.Series(ev)).all(axis=1).sum()) if ev else len(treino)
        consultas.append({'cenario': nome, 'evidencia': ev, 'n_treino_evidencia': suporte, **prob})
    # Um caso real do teste, escolhido por uma regra simples e documentada.
    indice = int(teste[teste.G2 == 'baixa'].sort_index().index[0])
    caso = teste.loc[indice].to_dict()
    evidencia = {c: caso[c] for c in COLUNAS if c != 'G3'}
    posterior = consultar(inf, evidencia)
    mb = sorted(modelo.get_markov_blanket('G3'))
    posterior_mb = consultar(inf, {c: evidencia[c] for c in mb})
    assert np.allclose(list(posterior.values()), list(posterior_mb.values()))
    sensibilidade = []
    for variavel in evidencia:
        reduzida = {k: v for k, v in evidencia.items() if k != variavel}
        sem = consultar(inf, reduzida)
        sensibilidade.append({'variavel': variavel, 'observado': evidencia[variavel],
            'p_baixa_sem': sem['baixa'], 'p_baixa_com': posterior['baixa'],
            'diferenca_pp': 100 * (posterior['baixa'] - sem['baixa'])})
    sensibilidade.sort(key=lambda x: abs(x['diferenca_pp']), reverse=True)
    # Compara estados de G2 mantendo as demais evidências. É associação, não intervenção.
    alternativas = {}
    for estado in ROTULOS:
        evidencia_alternativa = evidencia.copy()
        evidencia_alternativa['G2'] = estado
        alternativas[estado] = consultar(inf, evidencia_alternativa)
    resultados_teste = []
    for nome, m in modelos.items():
        inferencia = VariableElimination(m)
        probabilidades = []
        for _, registro in teste.iterrows():
            evidencia_teste = registro.drop('G3').to_dict()
            posterior_teste = consultar(inferencia, evidencia_teste)
            probabilidades.append([posterior_teste[estado] for estado in ROTULOS])
        p = np.asarray(probabilidades)
        y = [ROTULOS.index(s) for s in teste.G3]
        resultados_teste.append({'metodo': nome, 'acuracia': float(accuracy_score(y, p.argmax(axis=1))),
            'log_loss': float(log_loss(y, p, labels=[0, 1, 2]))})
    frequencias = treino.G3.value_counts(normalize=True).reindex(ROTULOS).values
    baseline = {'acuracia': float((teste.G3 == ROTULOS[int(frequencias.argmax())]).mean()),
                'log_loss': float(log_loss([ROTULOS.index(s) for s in teste.G3],
                                   np.tile(frequencias, (len(teste), 1)), labels=[0, 1, 2]))}
    for nome, m in modelos.items():
        chave = 'hc' if nome == 'Hill-Climbing' else 'chow_liu'
        (saida / f'cpts_{chave}.txt').write_text('\n\n'.join(str(c) for c in m.get_cpds()), encoding='utf-8')
        # Formato sem truncamento para leitura por programas.
        cpts = {}
        for cpd in m.get_cpds():
            cpts[cpd.variable] = {
                'variables': cpd.variables,
                'cardinality': cpd.cardinality.tolist(),
                'state_names': cpd.state_names,
                'values': cpd.values.tolist()
            }
        (saida / f'cpts_{chave}.json').write_text(json.dumps(cpts, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(consultas).drop(columns='evidencia').to_csv(saida / 'consultas.csv', index=False)
    pd.DataFrame(sensibilidade).to_csv(saida / 'sensibilidade.csv', index=False)
    pd.DataFrame(comparacoes).to_csv(saida / 'comparacao.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh([s['variavel'] for s in sensibilidade][::-1], [s['diferenca_pp'] for s in sensibilidade][::-1], color='#337d9d')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('P(G3 baixa | todas) - P(G3 baixa | sem variável), em p.p.')
    ax.set_title('Sensibilidade local por retirada de evidência')
    from analises import salvar_figura
    salvar_figura(fig, saida / 'sensibilidade.png')
    resultado = {'preprocessamento': resumo, 'comparacao': comparacoes, 'modelo_escolhido': escolhido,
                 'arestas': {n: sorted(m.edges()) for n, m in modelos.items()},
                 'comparacao_manual': {n: comparar(manual, m) for n, m in modelos.items()},
                 'consultas': consultas, 'teste': resultados_teste, 'baseline': baseline,
                 'caso': {'indice_original': indice, 'evidencia': evidencia, 'G3_real': caso['G3'],
                          'posterior': posterior, 'markov_blanket': mb, 'posterior_mb': posterior_mb,
                          'sensibilidade': sensibilidade, 'alternativas_G2': alternativas},
                 'ambiente': {'python': platform.python_version(),
                     **{m: importlib.metadata.version(m) for m in ['pgmpy','pandas','numpy','networkx','scikit-learn','matplotlib']}},
                 'sha256_dados': hashlib.sha256((BASE / 'dados/student-por.csv').read_bytes()).hexdigest()}
    (saida / 'resultados.json').write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    from analises import exportar_analises
    exportar_analises(saida, treino, teste, modelos, resultado, consultar)
    print(pd.DataFrame(comparacoes).to_string(index=False))
    print('\nModelo escolhido:', escolhido)
    print(pd.DataFrame(consultas).drop(columns='evidencia').to_string(index=False))
    print('\nAvaliação:', resultados_teste, '\nBaseline:', baseline)
    print('\nCaso:', resultado['caso'])
    return resultado


if __name__ == '__main__':
    executar()
