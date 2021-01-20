
import json
import re
from urllib.request import urlopen
import pandas as pd
# import seaborn as sns
import requests
import numpy as np
import dash_html_components as html
import plotly.io as pio
import plotly.graph_objects as go

from datetime import datetime
import plotly.express as px

from cxotime import CxoTime

import plotly.express as px

time_axis_format = [
    #         dict(dtickrange=[None, 600000], value="%H:%M:%S.%L\n"),
    dict(dtickrange=[None, 60000000], value="%H:%M:%S\n%Y:%j"),
    dict(dtickrange=[60000000, 315360000], value="%Y:%j"),
    dict(dtickrange=[315360000, "M1"], value="%e %b\n%Y:%j"),
    dict(dtickrange=["M1", "M6"], value="%Y:%j"),
    dict(dtickrange=["M6", None], value="%Y")
]

# font = 'Courier New, monospace'
font = 'Arial'

title_format = {
    'family': font,
    'size': 32,
    'color': '#666666',
    #     'color': '#7f7f7f'
}
sub_title_format = {
    'family': font,
    'size': 24,
    'color': '#666666'
}
axis_format = {
    'family': font,
    'size': 20,
    'color': '#666666'
}

label_format = {
    'family': font,
    'size': 24,
    'color': '#666666'
}

legend_format = {
    'family': font,
    'size': 16,
    'color': "#666666",
}

colors = px.colors.qualitative.D3


def hex_to_rgba(hexstr, opacity):
    hexstr = hexstr.lstrip('#')
    hlen = len(hexstr)
    rgba = [int(hexstr[i : i + int(hlen/3)], 16) for i in range(0, hlen, int(hlen/3))] + [opacity, ]
    return tuple(rgba)


def hex_to_rgba_str(hexstr, opacity):
    rgba = hex_to_rgba(hexstr, opacity)
    return f'rgba({rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]})'


def format_dates(cheta_dates):
    return np.array([datetime.strptime(d, '%Y:%j:%H:%M:%S.%f') for d in CxoTime(cheta_dates).date])


def format_plot_data(model_results, limit, state_data, dwell1_state, dwell2_state):
    #     keep_ind = find_non_repeated_points(msid_data[msid].vals)

    #     all_dates = format_dates(msid_data.times)

    plot_data = []

    plot_data.append({
        'type': 'scattergl',
        'x': format_dates(state_data['state_times']),
        'y': state_data['state_keys'],
        'name': 'State Keys',
        'line': {'color': '#666666', 'width': 2, 'shape': 'hv'},
        'mode': 'lines',
        'showlegend': False,
        'xaxis': 'x',
        'yaxis': 'y',
    })

    plot_data.append({
        'type': 'scattergl',
        'x': format_dates([state_data['state_times'][0], state_data['state_times'][-1]]),
        'y': [limit, limit],
        'name': 'State Keys',
        'line': {'color': 'black', 'width': 2, 'shape': 'hv', 'dash': 'dash'},
        'mode': 'lines',
        'showlegend': False,
        'xaxis': 'x2',
        'yaxis': 'y2',
    })

    plot_data.append({
        'type': 'scattergl',
        'x': format_dates(model_results.times),
        'y': model_results.mvals,
        'name': 'AACCCDPT Temperatures',
        'line': {'color': '#666666', 'width': 2, 'shape': 'hv'},
        'mode': 'lines',
        'showlegend': False,
        'xaxis': 'x2',
        'yaxis': 'y2',
    })

    return plot_data


def generate_converged_solution_plot_dict(plot_data, shapes, annotations, tstart, tstop, units='Celsius'):
    plot_start = datetime.strptime(CxoTime(tstart).date, '%Y:%j:%H:%M:%S.%f')
    plot_stop = datetime.strptime(CxoTime(tstop).date, '%Y:%j:%H:%M:%S.%f')

    plot_object = {
        'data': plot_data,
        'layout':
            {
                'hovermode': "closest",
                'autosize': False,
                'width': 1200,
                'height': 600,
                'margin': {'l': 80, 'r': 50, 't': 50, 'b': 70},
                'title':
                    {
                        'text': 'Converged Timbre Dwell Simulation',
                        'font': title_format,
                        'y': 0.98,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                'yaxis':
                    {
                        #                         'title':
                        #                             {
                        #                                 'text': 'Simulation Temperature',
                        #                                 'font': label_format
                        #                             },
                        'tickfont': axis_format,
                        'domain': [0.0, 0.19],
                        'range': [0.5, 2.5],
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'anchor': 'x',
                        'tickvals': [1, 2],
                        'ticktext': ['State 1', 'State 2'],
                    },

                'yaxis2':
                    {
                        'title':
                            {
                                'text': f'Simulation Temperature<br>({units})',
                                'font': label_format
                            },
                        'tickfont': axis_format,
                        'domain': [0.2, 1.0],
                        #                 'range': [-30, 45],
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'anchor': 'x2',
                    },

                'xaxis':
                    {
                        'domain': [0, 1],
                        'tickfont': axis_format,
                        'tickformatstops': time_axis_format,
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'range': [plot_start, plot_stop],
                        'showticklabels': True,
                        'anchor': 'y',
                    },

                'xaxis2':
                    {
                        'domain': [0, 1],
                        'tickfont': axis_format,
                        'tickformatstops': time_axis_format,
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'range': [plot_start, plot_stop],
                        'showticklabels': False,
                        'anchor': 'y2',
                        'matches': 'x',
                        'ticks': '',

                    },

                'showlegend': False,
                'template': 'simple_white',
                'shapes': shapes,
                'annotations': annotations,
            },
    }
    return plot_object


def format_shapes(state_data):
    shape_data = []

    s = zip(state_data['state_times'], state_data['state_keys'])
    for t1, s1 in s:
        t2, s2 = next(s)

        t3, s3 = next(s)  # Ignore
        t4, s4 = next(s)  # Ignore

        shape_data.extend([
            {
                'fillcolor': 'black',
                'line': {'width': 0},
                'opacity': 0.05,
                'type': 'rect',
                'x0': datetime.strptime(CxoTime(t1).date, '%Y:%j:%H:%M:%S.%f'),
                'x1': datetime.strptime(CxoTime(t2).date, '%Y:%j:%H:%M:%S.%f'),
                'y0': 0,
                'y1': 1,
                'xref': 'x2',
                'yref': 'y2 domain',

            },
            {
                'fillcolor': 'black',
                'line': {'width': 0},
                'opacity': 0.05,
                'type': 'rect',
                'x0': datetime.strptime(CxoTime(t1).date, '%Y:%j:%H:%M:%S.%f'),
                'x1': datetime.strptime(CxoTime(t2).date, '%Y:%j:%H:%M:%S.%f'),
                'y0': 0,
                'y1': 1,
                'xref': 'x',
                'yref': 'y domain',

            }
        ])
    return shape_data


def gen_unused_range(tstart, tstop, t_backoff=1725000):
    tstop = CxoTime(tstop).secs - t_backoff
    spans = [{
        'fillcolor': 'black',
        'line': {'width': 0},
        'opacity': 0.25,
        'type': 'rect',
        'x0': datetime.strptime(CxoTime(tstart).date, '%Y:%j:%H:%M:%S.%f'),
        'x1': datetime.strptime(CxoTime(tstop).date, '%Y:%j:%H:%M:%S.%f'),
        'y0': 0,
        'y1': 1,
        'xref': 'x',
        'yref': 'y domain',

    },
        {
            'fillcolor': 'black',
            'line': {'width': 0},
            'opacity': 0.25,
            'type': 'rect',
            'x0': datetime.strptime(CxoTime(tstart).date, '%Y:%j:%H:%M:%S.%f'),
            'x1': datetime.strptime(CxoTime(tstop).date, '%Y:%j:%H:%M:%S.%f'),
            'y0': 0,
            'y1': 1,
            'xref': 'x2',
            'yref': 'y2 domain',

        }
    ]
    return spans


def gen_range_annotations(tstart, tstop, yloc1, yloc2, t_backoff=1725000):
    tstart = CxoTime(tstart).secs
    tstop = CxoTime(tstop).secs
    tmid = tstop - t_backoff

    ttext = (tstop + tmid) / 2.
    arrow1 = {
        'x': datetime.strptime(CxoTime(tmid).date, '%Y:%j:%H:%M:%S.%f'),
        'y': yloc1,
        'text': '',
        'showarrow': True,
        'arrowhead': 2,
        'arrowwidth': 3,
        'arrowcolor': 'rgb(100,100,100)',
        'xref': "x2",
        'yref': "y2",
        'ax': datetime.strptime(CxoTime(ttext).date, '%Y:%j:%H:%M:%S.%f'),
        'ay': yloc1,
        'axref': 'x2',
        'ayref': 'y2',
        'xanchor': "center",
        'yanchor': "bottom",
        'font': label_format
    }
    arrow2 = {
        'x': datetime.strptime(CxoTime(tstop).date, '%Y:%j:%H:%M:%S.%f'),
        'y': yloc1,
        'text': '',
        'showarrow': True,
        'arrowhead': 2,
        'arrowwidth': 3,
        'arrowcolor': 'rgb(100,100,100)',
        'xref': "x2",
        'yref': "y2",
        'ax': datetime.strptime(CxoTime(ttext).date, '%Y:%j:%H:%M:%S.%f'),
        'ay': yloc1,
        'axref': 'x2',
        'ayref': 'y2',
        'xanchor': "center",
        'yanchor': "bottom",
        'font': label_format
    }
    text = {
        'x': datetime.strptime(CxoTime(ttext).date, '%Y:%j:%H:%M:%S.%f'),
        'y': yloc2,
        'text': 'Data Range used for evaluation',
        'showarrow': False,
        'xref': "x2",
        'yref': "y2",
        'xanchor': "center",
        'yanchor': "bottom",
        'font': label_format
    }
    annotations = [arrow1, arrow2, text]

    return annotations


def gen_limit_annotation(xloc, yloc, limit, units):
    text_dict = {
        'x': datetime.strptime(CxoTime(xloc).date, '%Y:%j:%H:%M:%S.%f'),
        'y': yloc,
        'text': f'Limit = {limit} {units}',
        'showarrow': False,
        'xref': "x2",
        'yref': "y2",
        'xanchor': "center",
        'yanchor': "bottom",
        'font': label_format
    }
    return [text_dict, ]


def gen_shading_annotation(xloc, yloc, dwell1_text, dwell2_text):
    text1 = f'Lightly Shaded Vertical Bands = Dwell State #1 ({dwell1_text})'
    text2 = f'Unshaded Vertical Bands = Dwell State #2 ({dwell2_text})'
    text = f'{text1}<br>{text2}'

    text_dict = {
        'x': datetime.strptime(CxoTime(xloc).date, '%Y:%j:%H:%M:%S.%f'),
        'y': yloc,
        'text': text,
        'showarrow': False,
        'xref': "x2",
        'yref': "y2",
        'xanchor': "right",
        'yanchor': "bottom",
        'font': label_format,
        'align': 'right'
    }
    return [text_dict, ]


def generate_example_balance_plot_dict(t_dwell1, t_dwell2, dwell1_state, dwell2_state):

    plot_object = {
        'data':  {
            'x': ['Hot Dwell', 'Cold Dwell'],
            'y': [np.round(t_dwell1, 1), t_dwell2, 1],
            'type': 'bar',
            'text': [f'Dwell #1 State<br>Pitch: {dwell1_state["pitch"]}<br><br>Dwell Time: {np.round(t_dwell1, 0):.0f}s<br>(Input)',
                     f'Dwell #2 State<br>Pitch: {dwell2_state["pitch"]}<br><br>Dwell Time: {np.round(t_dwell2, 0):.0f}s<br>(Calculated)'],
            'textposition': 'inside',
            'textfont': {'family': font, 'size': 24, 'color': 'white'},
            'marker': {'color': [colors[3], colors[0]]}
        },
        'layout':
            {
                'hovermode': "closest",
                'autosize': False,
                'width': 1200,
                'height': 600,
                'margin': {'l': 80, 'r': 50, 't': 50, 'b': 70},
                'title':
                    {
                        'text': 'Timbre Produced Dwell Balance',
                        'font': title_format,
                        'y': 0.98,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                'yaxis':
                    {
                        'title':{'text': 'Dwell Time (Kiloseconds)','font': label_format},
                        'tickfont': axis_format,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'anchor': 'x',
                    },
                'xaxis':
                    {
                        'domain': [0, 1],
                        'tickfont': axis_format,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'showticklabels': True,
                        'anchor': 'y',
                    },


                'showlegend': False,
                'template': 'simple_white',
            },
    }
    return plot_object


def generate_step_2_plot_dict(plot_data, tstart, tstop, title, units='Celsius'):
    plot_start = datetime.strptime(CxoTime(tstart).date, '%Y:%j:%H:%M:%S.%f')
    plot_stop = datetime.strptime(CxoTime(tstop).date, '%Y:%j:%H:%M:%S.%f')

    plot_object = {
        'data': plot_data,
        'layout':
            {
                'hovermode': "closest",
                'autosize': False,
                'width': 1200,
                'height': 600,
                'margin': {'l': 80, 'r': 50, 't': 50, 'b': 70},
                'title':
                    {
                        'text': title,
                        'font': title_format,
                        'y': 0.98,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                'yaxis':
                    {
                        'title':
                        {
                             'text': f'Resulting Temperatures for<br>Dwell #2 Guesses ({units})',
                             'font': label_format
                        },
                        'tickfont': axis_format,
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'anchor': 'x',
                    },

                'xaxis':
                    {
                        'domain': [0, 1],
                        'tickfont': axis_format,
                        'tickformatstops': time_axis_format,
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'range': [plot_start, plot_stop],
                        'showticklabels': True,
                        'tickangle': 30,
                        'anchor': 'y',
                    },

                'showlegend': True,
                'template': 'simple_white',
            },
    }
    return plot_object


def format_step_2_plot_data(model_data, limit, tstart, tstop):
    plot_start = datetime.strptime(CxoTime(tstart).date, '%Y:%j:%H:%M:%S.%f')
    plot_stop = datetime.strptime(CxoTime(tstop).date, '%Y:%j:%H:%M:%S.%f')

    seq_colors = px.colors.n_colors(hex_to_rgba_str(colors[3], 1),
                                    hex_to_rgba_str(colors[0], 1),
                                    len(model_data),
                                    colortype='rgb')
    plot_data = []

    for (t, results), c in zip(model_data.items(), seq_colors):
        model_results = results['model_results']['aacccdpt']

        plot_data.append({
            'type': 'scattergl',
            'x': format_dates(model_results.times),
            'y': model_results.mvals,
            'name': f't_dwell2 = {t:.1f}',
            'line': {'color': c, 'width': 2, 'shape': 'hv'},
            'mode': 'lines',
            'showlegend': True,
            'xaxis': 'x',
            'yaxis': 'y',
        })

    plot_data.append({
        'type': 'scattergl',
        'x': [plot_start, plot_stop],
        'y': [limit, limit],
        'name': 'Limit',
        'line': {'color': 'black', 'width': 2, 'shape': 'hv', 'dash': 'dash'},
        'mode': 'lines',
        'showlegend': False,
        'xaxis': 'x',
        'yaxis': 'y',
    })

    return plot_data


def generate_step_3_max_temp_plot_dict(output, title, t_dwell2, units='Celsius'):
    seq_colors = px.colors.n_colors(hex_to_rgba_str(colors[3], 1),
                                    hex_to_rgba_str(colors[0], 1),
                                    len(output),
                                    colortype='rgb')

    plot_object = {
        'data': [
            {
                'type': 'scattergl',
                'x': output['duration2'],
                'y': output['max'],
                'name': f'Dwell 2 Duration Guesses',
                'line': {'color': 'black', 'width': 2, 'shape': 'hv'},
                'marker': {
                    'size': 24,
                    'cmax': max(output['duration2']),
                    'cmin': min(output['duration2']),
                    'color': output['duration2'],
                    'colorscale': list(zip(np.linspace(0, 1, 10), seq_colors)),
                },
                'mode': 'markers',
                'showlegend': False,
                'xaxis': 'x',
                'yaxis': 'y',
            }
        ],
        'layout':
            {
                'hovermode': "closest",
                'autosize': False,
                'width': 1200,
                'height': 600,
                'margin': {'l': 80, 'r': 50, 't': 50, 'b': 70},
                'title':
                    {
                        'text': title,
                        'font': title_format,
                        'y': 0.98,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                'yaxis':
                    {
                        'title':
                            {
                                'text': f'Temperature ({units})',
                                'font': label_format
                            },
                        'tickfont': axis_format,
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'anchor': 'x',
                    },

                'xaxis':
                    {
                        'domain': [0, 1],
                        'tickfont': axis_format,
                        #                         'tickformatstops': time_axis_format,
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        #                         'range': [plot_start, plot_stop],
                        'showticklabels': True,
                        #                         'tickangle': 30,
                        'anchor': 'y',
                    },

                'showlegend': True,
                'template': 'simple_white',
                'shapes': [
                    {
                        'line': {'color': 'black', 'dash': 'dash', 'width': 3},
                        'opacity': 0.65,
                        'type': 'line',
                        'x0': 0,
                        'x1': 1,
                        'y0': -6.5,
                        'y1': -6.5,
                        'xref': 'x domain',
                        'yref': 'y',
                    },

                    {
                        'line': {'color': 'black', 'dash': 'dash', 'width': 3},
                        'opacity': 0.65,
                        'type': 'line',
                        'x0': t_dwell2,
                        'x1': t_dwell2,
                        'y0': 0,
                        'y1': 1,
                        'xref': 'x',
                        'yref': 'y domain',
                    },

                ],
                'annotations': [
                    {
                        'x': 65000,
                        'y': -6.485,
                        'text': 'Limit = -6.5C',
                        'showarrow': False,
                        'xref': "x",
                        'yref': "y",
                        'xanchor': "center",
                        'yanchor': "bottom",
                        'font': label_format
                    },
                    {
                        'x': t_dwell2 - 200,
                        'y': -7.15,
                        'text': f't_dwell2 = {t_dwell2:.0f}s',
                        'showarrow': False,
                        'xref': "x",
                        'yref': "y",
                        'xanchor': "right",
                        'yanchor': "bottom",
                        'font': label_format,
                        'textangle': -90,
                    },
                ],
            },
    }
    return plot_object


def generate_timbre_dwell_plot_data(results, filter_set):
    plot_data = []

    for plot_set in filter_set:

        ind = np.zeros(len(results)) < 1
        name = []
        for key, value in plot_set.items():
            ind = ind & (results[key] == value)
            name.append(f'{str(key).capitalize()}: {value}')

        if np.median(results['hotter_state'][ind & results['converged']]) == 1:
            plot_color = colors[0]
        else:
            plot_color = colors[3]

        plot_data.append(
            {
                'type': 'scattergl',
                'x': results['pitch2'][ind],
                'y': results['t_dwell2'][ind],
                'name': ' '.join(name),
                'line': {'color':plot_color, 'width': 2},
                'marker': {
                    'size': 10,
                    'color': plot_color,
                },
                'mode': 'lines+markers',
                # 'showlegend': legend,
                'xaxis': 'x',
                'yaxis': 'y',
            }
        )

    return plot_data


def generate_timber_output_plot_dict(plot_data, title, legend=True):
    plot_object = {
        'data': plot_data,
        'layout':
            {
                'hovermode': "closest",
                'autosize': False,
                'width': 1200,
                'height': 600,
                'margin': {'l': 80, 'r': 50, 't': 50, 'b': 70},
                'title':
                    {
                        'text': title,
                        'font': title_format,
                        'y': 0.98,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                'yaxis':
                    {
                        'title':
                            {
                                'text': f'Dwell Time (s)',
                                'font': label_format
                            },
                        'tickfont': axis_format,
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'anchor': 'x',
                        'showgrid': True,
                    },

                'xaxis':
                    {
                        'domain': [0, 1],
                        'tickfont': axis_format,
                        #  'tickformatstops': time_axis_format,
                        'zeroline': False,
                        'linecolor': '#666666',
                        'linewidth': 1,
                        'mirror': True,
                        'range': [45, 180],
                        'showticklabels': True,
                        # 'tickangle': 30,
                        'anchor': 'y',
                        'showgrid': True,
                    },

                'showlegend': legend,
                'template': 'simple_white',
            },
    }
    return plot_object

