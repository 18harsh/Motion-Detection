from motionDetector import df
# import pandas as pd
# df = pd.read_csv("Times.csv")
from datetime import datetime
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource

# df["Start"]= df["Start"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f"))
# df["End"]= df["End"].apply(lambda y: datetime.strptime(y,"%Y-%m-%d %H:%M:%S.%f"))

df["Start_string"]=df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["End_string"]=df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")

cds=ColumnDataSource(df)

p=figure(x_axis_type='datetime', height=500 ,width=1000 ,title="Motion Graph",sizing_mode="scale_width")

p.yaxis.minor_tick_line_color = None
# p.ygrid[0].ticker.desired_num_ticks = 1

hover=HoverTool(tooltips=[("Start","@Start_string"),("End","@End_string")])
p.add_tools(hover)

q=p.quad(left="Start",right="End",bottom=0,top=1,color="green",source=cds)
output_file("Graph.html")

show(p)