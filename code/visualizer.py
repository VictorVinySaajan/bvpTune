import plotly.express as px

class Visualizer():
    def __init__(self):
        """Init function"""
        
    def visualize2d(self, dataframe):
        fig = px.scatter(dataframe, x=dataframe.columns[-1], y=dataframe.columns[-2],
                        custom_data=[dataframe.columns[0], dataframe.columns[1], dataframe.columns[2],
                                    dataframe.columns[3], dataframe.columns[4], dataframe.columns[5],
                                    dataframe.columns[6], dataframe.columns[7], dataframe.columns[8]],
                        width=500, height=400)
        fig.update_traces(
            hovertemplate="<br>".join([
            "test_case_type: %{customdata[0]}",
            "max_grid_points: %{customdata[1]}",
            "newton_critical_tolerance: %{customdata[2]}",
            "newton_armijo_probes: %{customdata[3]}",
            "newton_max_iterations: %{customdata[4]}",
            "newton_tolerance: %{customdata[5]}",
            "add_factor: %{customdata[6]}",
            "remove_factor: %{customdata[7]}",
            "use_collocation_scaling: %{customdata[8]}",
            ])
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="LightSteelBlue",
        )
        
        return fig.show()
        
    def visualize3d(self, dataframe):
        fig = px.scatter_3d(dataframe, x=dataframe.columns[-1], y=dataframe.columns[-2], z=dataframe.columns[-3],
                          custom_data=[dataframe.columns[0], dataframe.columns[1], dataframe.columns[2],
                                    dataframe.columns[3], dataframe.columns[4], dataframe.columns[5],
                                    dataframe.columns[6], dataframe.columns[7], dataframe.columns[8]],
                        width=500, height=400)
        
        fig.update_traces(
            hovertemplate="<br>".join([
            "test_case_type: %{customdata[0]}",
            "max_grid_points: %{customdata[1]}",
            "newton_critical_tolerance: %{customdata[2]}",
            "newton_armijo_probes: %{customdata[3]}",
            "newton_max_iterations: %{customdata[4]}",
            "newton_tolerance: %{customdata[5]}",
            "add_factor: %{customdata[6]}",
            "remove_factor: %{customdata[7]}",
            "use_collocation_scaling: %{customdata[8]}",
            ])
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="LightSteelBlue",
        )
        
        return fig.show()