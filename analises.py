"""Exportações para a análise do grupo e montagem das figuras do relatório."""
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def salvar_figura(figura, destino):
    """PNG para visualização e SVG/PDF para inserir sem perder definição."""
    destino = Path(destino)
    figura.tight_layout()
    for extensao in ['png', 'svg', 'pdf']:
        figura.savefig(destino.with_suffix('.' + extensao), dpi=300, bbox_inches='tight')
    plt.close(figura)


def salvar_csv(tabela, destino):
    # BOM ajuda o Excel do Windows a reconhecer os acentos.
    tabela.to_csv(destino, index=False, encoding='utf-8-sig')


def exportar_analises(saida, treino, teste, modelos, resultado, consultar):
    """Gera arquivos complementares sem mudar o ajuste dos modelos."""
    from pgmpy.inference import VariableElimination

    estados = ['baixa', 'media', 'alta']
    cores = ['#c55e54', '#d6a441', '#398694']
    figuras = saida / 'figuras'
    tabelas = saida / 'tabelas'
    figuras.mkdir(exist_ok=True)
    tabelas.mkdir(exist_ok=True)

    cortes = resultado['preprocessamento']['cortes']
    salvar_csv(pd.DataFrame([
        {'variavel': c, 'minimo_treino': v[0], 'quantil_1_3': v[1],
         'quantil_2_3': v[2], 'maximo_treino': v[3],
         'regra': 'baixa <= q1; media > q1 e <= q2; alta > q2'}
        for c, v in cortes.items()]), tabelas / 'limites_discretizacao.csv')
    frequencias = []
    for conjunto, dados in [('treino', treino), ('teste', teste)]:
        for c in dados:
            for estado, n in dados[c].value_counts().sort_index().items():
                frequencias.append({'conjunto': conjunto, 'variavel': c, 'estado': estado,
                                    'quantidade': int(n), 'percentual': 100 * n / len(dados)})
    salvar_csv(pd.DataFrame(frequencias), tabelas / 'frequencias_categorias.csv')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    for i, (nome, dados) in enumerate([('Treino', treino), ('Teste', teste)]):
        p = dados.G3.value_counts(normalize=True).reindex(estados, fill_value=0) * 100
        ax.bar(x + (i - .5) * .35, p, width=.35, label=nome)
    ax.set(xticks=x, xticklabels=estados, ylabel='Registros (%)', title='Distribuição das faixas de G3')
    ax.legend(); salvar_figura(fig, figuras / 'distribuicao_g3')

    consultas = pd.DataFrame(resultado['consultas'])
    salvar_csv(consultas.assign(evidencia=consultas.evidencia.map(
        lambda e: json.dumps(e, ensure_ascii=False))), tabelas / 'consultas_com_evidencias.csv')
    fig, ax = plt.subplots(figsize=(11, 5))
    fundo = np.zeros(len(consultas))
    for estado, cor in zip(estados, cores):
        valores = consultas[estado].to_numpy() * 100
        ax.barh(consultas.cenario, valores, left=fundo, color=cor, label=estado)
        fundo += valores
    ax.set(xlim=(0, 100), xlabel='Probabilidade de G3 (%)', title='Inferência com diferentes evidências')
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3)
    salvar_figura(fig, figuras / 'consultas_g3')

    comparacao = pd.DataFrame(resultado['comparacao'])
    salvar_csv(comparacao, tabelas / 'comparacao_estruturas.csv')
    arestas = []
    for nome, modelo in modelos.items():
        for u, v in sorted(modelo.edges()):
            arestas.append({'metodo': nome, 'origem': u, 'destino': v})
        # As variáveis e seus estados ficam explícitos no JSON de CPTs original.
        cpts = []
        for cpd in modelo.get_cpds():
            variaveis = cpd.variables
            for combinacao in itertools.product(*(cpd.state_names[v] for v in variaveis)):
                valores = dict(zip(variaveis, combinacao))
                pais = {v: valores[v] for v in variaveis[1:]}
                cpts.append({'variavel': cpd.variable, 'estado': valores[cpd.variable],
                             'pais': json.dumps(pais, ensure_ascii=False, sort_keys=True),
                             'probabilidade': float(cpd.get_value(**valores))})
        chave = 'hc' if nome == 'Hill-Climbing' else 'chow_liu'
        salvar_csv(pd.DataFrame(cpts), tabelas / f'cpts_{chave}.csv')
    salvar_csv(pd.DataFrame(arestas), tabelas / 'arestas_aprendidas.csv')
    linhas_comparacao = []
    for metodo, info in resultado['comparacao_manual'].items():
        for tipo in ['mesma_direcao', 'invertidas', 'ausentes', 'novas']:
            for a, b in info[tipo]:
                linhas_comparacao.append({'metodo': metodo, 'situacao': tipo, 'no_1': a, 'no_2': b})
    salvar_csv(pd.DataFrame(linhas_comparacao), tabelas / 'comparacao_com_manual.csv')

    predicoes = []
    for nome, modelo in modelos.items():
        inf = VariableElimination(modelo)
        reais, previstas = [], []
        for indice, registro in teste.iterrows():
            evidencia = registro.drop('G3').to_dict()
            posterior = consultar(inf, evidencia)
            prevista = max(estados, key=lambda s: posterior[s])
            reais.append(registro.G3); previstas.append(prevista)
            predicoes.append({'metodo': nome, 'indice_original': indice, 'real': registro.G3,
                              'prevista': prevista, **{'p_' + k: v for k, v in posterior.items()}})
        matriz = confusion_matrix(reais, previstas, labels=estados)
        chave = 'hc' if nome == 'Hill-Climbing' else 'chow_liu'
        salvar_csv(pd.DataFrame(matriz, columns=['prevista_' + s for s in estados]).assign(
            real=estados), tabelas / f'matriz_confusao_{chave}.csv')
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matriz, cmap='Blues', vmin=0)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matriz[i, j]), ha='center', va='center',
                        color='white' if matriz[i, j] > matriz.max() / 2 else 'black')
        ax.set(xticks=range(3), xticklabels=estados, yticks=range(3), yticklabels=estados,
               xlabel='Faixa prevista', ylabel='Faixa observada', title=f'Matriz de confusão — {nome}')
        fig.colorbar(im, ax=ax, label='Registros')
        salvar_figura(fig, figuras / f'matriz_confusao_{chave}')
    salvar_csv(pd.DataFrame(predicoes), tabelas / 'predicoes_teste.csv')
    metricas = pd.DataFrame(resultado['teste'] + [{'metodo': 'Baseline marginal', **resultado['baseline']}])
    salvar_csv(metricas, tabelas / 'metricas_teste.csv')
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, coluna, titulo in zip(eixos, ['acuracia', 'log_loss'],
                                 ['Acurácia (maior é melhor)', 'Log-loss (menor é melhor)']):
        ax.bar(metricas.metodo, metricas[coluna], color=['#398694', '#5277a3', '#999999'])
        ax.set_title(titulo); ax.tick_params(axis='x', labelrotation=20)
        ax.set_ylim(bottom=0)
    salvar_figura(fig, figuras / 'avaliacao_teste')

    caso = resultado['caso']
    alternativas = pd.DataFrame(caso['alternativas_G2']).T.rename_axis('G2').reset_index()
    salvar_csv(alternativas, tabelas / 'cenarios_alternativos_g2.csv')
    salvar_csv(pd.DataFrame([{'indice_original': caso['indice_original'],
        **caso['evidencia'], 'G3_real': caso['G3_real'],
        **{'p_' + k: v for k, v in caso['posterior'].items()}}]), tabelas / 'caso_explicado.csv')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (estado, cor) in enumerate(zip(estados, cores)):
        ax.bar(x + (i - 1) * .24, alternativas[estado] * 100, width=.24, label=estado, color=cor)
    ax.set(xticks=x, xticklabels=estados, xlabel='Estado observado de G2',
           ylabel='Probabilidade de G3 (%)', title='Cenários condicionais para o caso explicado')
    ax.legend(title='G3'); salvar_figura(fig, figuras / 'cenarios_alternativos_g2')
    escolhido = modelos[resultado['modelo_escolhido']]
    nos = sorted(escolhido.nodes())
    grafo = nx.DiGraph(); grafo.add_nodes_from(nos); grafo.add_edges_from(escolhido.edges())
    pos = nx.spring_layout(grafo, seed=42, k=1.2)
    fig, ax = plt.subplots(figsize=(8, 5))
    for no, (px, py) in pos.items():
        cor = '#e5af54' if no == 'G3' else '#7fb8c5' if no in caso['markov_blanket'] else '#eeeeee'
        ax.text(px, py, no, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=.5', fc=cor, ec='#456575'))
    for u, v in grafo.edges():
        ax.annotate('', xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle='->', shrinkA=23, shrinkB=23, color='#456575'))
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), title='G3 (laranja) e seu cobertor de Markov (azul)')
    ax.axis('off'); salvar_figura(fig, figuras / 'cobertor_markov_g3')
