import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import warnings
import os
from statsbombpy import sb
pd.set_option('display.max_columns', None)

import joblib  # Para guardar el modelo
import shap

warnings.filterwarnings("ignore")

model_cols = ['Ball Receipt*', 'Ball Recovery', 'Block', 'Carry', 'Dribble', 'Duel', 'Pass']
def prepare_model_data(match_id, team_name=None):
    # Obtener los eventos del partido usando match_id
    match_data = sb.events(match_id=match_id)
    
    # Obtener la lista de equipos participantes en el partido
    teams_in_match = match_data['team'].unique()
    
    # Si se proporciona un equipo, verificar si está en el partido
    if team_name:
        if team_name not in teams_in_match:
            return f"El equipo '{team_name}' no jugó este partido."
    
    # Especificar los tipos de eventos a incluir (solo los indicados en model_cols)
    model_cols = ['Ball Receipt*', 'Ball Recovery', 'Block', 'Carry', 'Dribble', 'Duel', 'Pass']
    
    # Filtrar los eventos para incluir solo los tipos especificados
    events_df = match_data[match_data['type'].isin(model_cols)].copy()
    
    # Si se proporciona un equipo, filtrar los eventos por equipo
    if team_name:
        events_df = events_df[events_df['team'] == team_name].copy()

    # Obtener los eventos de tiros para calcular las métricas de xG
    shot_events = match_data[match_data['type'] == 'Shot'].copy()
    if team_name:
        shot_events = shot_events[shot_events['team'] == team_name].copy()
    
    # Calcular total de xG por jugador y equipo
    xg_df = (
        shot_events
        .groupby(['player', 'team'])['shot_statsbomb_xg']
        .sum()
        .reset_index(name='xG')
    )
    
    # Calcular número de goles por jugador (donde 'shot_outcome' == 'Goal')
    goals_df = (
        shot_events[shot_events['shot_outcome'] == 'Goal']
        .groupby(['player', 'team'])
        .size()
        .reset_index(name='Goals')
    )
    
    # Calcular total de tiros por jugador
    total_shots = (
        shot_events
        .groupby(['player', 'team'])
        .size()
        .reset_index(name='Shot')
    )
    
    # Crear un DataFrame para contener las estadísticas de tiro
    shot_stats_df = total_shots.merge(xg_df, on=['player', 'team'], how='left').fillna({'xG': 0})
    shot_stats_df = shot_stats_df.merge(goals_df, on=['player', 'team'], how='left').fillna({'Goals': 0})
    
    # Calcular la sobreperformance de xG
    shot_stats_df['xG Overperformance'] = shot_stats_df['Goals'] - shot_stats_df['xG']
    
    # Calcular xG por tiro
    shot_stats_df['xG per Shot'] = (shot_stats_df['xG'] / shot_stats_df['Shot']).fillna(0)
    
    # Calcular tiros a puerta (Shots on Target)
    shots_on_target = shot_events[shot_events['shot_outcome'].isin(['Goal', 'Saved'])]
    shots_on_target_count = shots_on_target.groupby(['player', 'team']).size().reset_index(name='Shots on Target')
    
    # Unir tiros a puerta al DataFrame de estadísticas de tiro
    shot_stats_df = shot_stats_df.merge(shots_on_target_count, on=['player', 'team'], how='left').fillna({'Shots on Target': 0})
    
    # Calcular precisión de tiro (Shot Accuracy %)
    shot_stats_df['Shot Accuracy %'] = (shot_stats_df['Shots on Target'] / shot_stats_df['Shot']) * 100
    shot_stats_df['Shot Accuracy %'] = shot_stats_df['Shot Accuracy %'].fillna(0)
    
    # Calcular grandes ocasiones falladas (xG >= 0.3)
    big_chances_missed = shot_events[
        (shot_events['shot_statsbomb_xg'] >= 0.3) &
        (shot_events['shot_outcome'] != 'Goal')
    ]
    big_chances_missed_count = big_chances_missed.groupby(['player', 'team']).size().reset_index(name='Big Chances Missed')
    
    # Unir grandes ocasiones falladas al DataFrame de estadísticas de tiro
    shot_stats_df = shot_stats_df.merge(big_chances_missed_count, on=['player', 'team'], how='left').fillna({'Big Chances Missed': 0})
    
    # Crear una tabla pivote para contar los eventos de interés por jugador y equipo (solo model_cols)
    events_summary_df = (
        events_df.pivot_table(
            index=['player', 'team'],
            columns='type',
            aggfunc='size',
            fill_value=0
        )
        .reset_index()
    )
    
    # Unir las estadísticas de tiro con el resumen de eventos
    events_df = events_summary_df.merge(shot_stats_df, on=['player', 'team'], how='left').fillna(0)
    
    # Redondear todas las columnas numéricas a 4 decimales
    events_df = events_df.round(4)
    
    # Devolver el DataFrame final con solo los eventos de model_cols y las estadísticas de xG
    return events_df


def output_shaps(id_partido_1, team_1, id_partido_2, team_2):
    '''
    
    '''
    model_cols = ['Ball Receipt*', 'Ball Recovery', 'Block', 'Carry',
              'Dribble', 'Duel', 'Pass', 
              ]
    target = ['xG']
    loaded_scaler = joblib.load('scaler_model.pkl')
    best_rf = joblib.load('best_gradient_boosting_model.pkl')
    explainer = shap.TreeExplainer(best_rf)
    t1 = prepare_model_data(id_partido_1, team_1)
    t2 = prepare_model_data(id_partido_2, team_2)

    events_df = pd.concat([t1, t2])

    X_scaled = loaded_scaler.transform(events_df[model_cols])

    # Calcular los valores SHAP
    shap_values = explainer.shap_values(X_scaled)

    # Crear un DataFrame con los valores SHAP
    shap_df = pd.DataFrame(shap_values, columns=[f'sh_{col}' for col in model_cols])

    # Calcular la media de los valores SHAP para cada fila y agregarla como una nueva columna
    shap_df['shapley'] = shap_df.mean(axis=1)

    # Hacer predicciones con el modelo cargado y agregar la columna y_pred al DataFrame original
    shap_df['y_pred'] = best_rf.predict(X_scaled)

    # Unir el DataFrame original con el DataFrame de SHAP
    final_df = pd.concat([events_df.reset_index(drop=True), shap_df], axis=1)

    # Filtrar el DataFrame para incluir sólo las columnas usadas en el modelo, SHAP values, el target y y_pred
    columns_to_keep = ['player', 'team'] + model_cols + list(shap_df.columns) + target
    final_df_filtered = final_df[columns_to_keep]

    sh_team_1 = final_df_filtered[final_df_filtered['team'] == team_1]['shapley'].mean()
    sh_team_2 =  final_df_filtered[final_df_filtered['team'] == team_2]['shapley'].mean()
    sh_glb = final_df_filtered['shapley'].mean()

    final_df_filtered['plus_' + team_1[:3]] = final_df_filtered['shapley'] >= sh_team_1 
    final_df_filtered['plus_' + team_2[:3]] = final_df_filtered['shapley'] >= sh_team_2
    final_df_filtered['plus_glb'] = final_df_filtered['shapley'] >= sh_glb

    return final_df_filtered



st.set_page_config(page_title="Análisis de Partidos de Fútbol", layout="wide")

# Título
st.title("Análisis de Partidos de Fútbol con SHAP y xG")

# Diccionario de partidos
dict_matches = {'arg_col_final': 3943077, 
                'can_uru_3rd_place_final': 3943076, 
                'uru_col_semi-finals': 3942852, 
                'arg_can_semi-finals': 3942785, 
                'col_pan_quarter-finals': 3942416, 
                'uru_bra_quarter-finals': 3942415,
                'ven_can_quarter-finals': 3942229, 
                'arg_ecu_quarter-finals': 3942228}

# Selección de partidos y equipos
match_options = list(dict_matches.keys())
selected_match_1 = st.selectbox("Selecciona el primer partido", match_options)
selected_match_2 = st.selectbox("Selecciona el segundo partido", match_options)
# Función auxiliar para obtener equipos de un partido
def obtener_equipos(match_id):
    # Obtener los datos del partido y los equipos únicos
    match_data = sb.events(match_id=match_id)
    equipos = match_data['team'].unique()
    return equipos

# Generar opciones de equipo basadas en el partido seleccionado
equipos_match_1 = obtener_equipos(dict_matches[selected_match_1])
equipos_match_2 = obtener_equipos(dict_matches[selected_match_2])

# Selección de equipos en listas desplegables
selected_team_1 = st.selectbox("Selecciona el equipo 1", equipos_match_1)
selected_team_2 = st.selectbox("Selecciona el equipo 2", equipos_match_2)

# Botón para generar datos
if st.button("Generar Análisis"):
    match_id_1 = dict_matches[selected_match_1]
    match_id_2 = dict_matches[selected_match_2]
    
    try:
        # Llamada a la función para procesar datos
        final_df_filtered = output_shaps(match_id_1, selected_team_1, match_id_2, selected_team_2)
        
        # Mostrar datos generales del análisis
        st.write("Resultados del análisis:")
        st.dataframe(final_df_filtered)
        
        # Visualización con Plotly
        fig = px.scatter(final_df_filtered, x='xG', y='shapley', color='team', hover_data=['player'])
        fig.update_layout(title="SHAP vs xG por Jugador", xaxis_title="xG", yaxis_title="SHAP")
        st.plotly_chart(fig, use_container_width=True)
        
        # Filtrar jugadores destacados para cada equipo
        team1_df = final_df_filtered[(final_df_filtered['team'] == selected_team_1) & (final_df_filtered[f'plus_{selected_team_1[:3]}'] == True)][['player', 'team', 'shapley'] + model_cols]
        team2_df = final_df_filtered[(final_df_filtered['team'] == selected_team_2) & (final_df_filtered[f'plus_{selected_team_2[:3]}'] == True)][['player', 'team', 'shapley'] + model_cols]
        
        # Mostrar resultados en dos columnas
        st.write("### Jugadores destacados según el análisis de SHAP")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Jugadores destacados de {selected_team_1}")
            st.dataframe(team1_df[['player', 'team', 'shapley']])
        
        with col2:
            st.write(f"Jugadores destacados de {selected_team_2}")
            st.dataframe(team2_df[['player', 'team', 'shapley']])
        
        # Diagrama de radar para los jugadores destacados
        st.write("### Comparación de Jugadores Destacados en Radar")

        # Preparación de datos para el radar
        radar_data_team1 = team1_df.groupby('player')[model_cols].mean().reset_index()
        radar_data_team2 = team2_df.groupby('player')[model_cols].mean().reset_index()

        radar_fig = go.Figure()

        # Añadir datos del equipo 1
        for _, row in radar_data_team1.iterrows():
            radar_fig.add_trace(go.Scatterpolar(
                r=row[model_cols].values,
                theta=model_cols,
                fill='toself',
                name=f"{row['player']} ({selected_team_1})"
            ))

        # Añadir datos del equipo 2
        for _, row in radar_data_team2.iterrows():
            radar_fig.add_trace(go.Scatterpolar(
                r=row[model_cols].values,
                theta=model_cols,
                fill='toself',
                name=f"{row['player']} ({selected_team_2})"
            ))

        # Configurar diseño del gráfico de radar
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            showlegend=True,
            title="Comparación de Métricas de Jugadores Destacados"
        )

        st.plotly_chart(radar_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")