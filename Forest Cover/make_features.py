import numpy as np
import pandas as pd
import scipy as sp


# Because we had issues with underfitting the model, we rely on this method 
# to create new features that will be of use later in the training process
# they are made mostly through exploratory analysis(scatter plots, histograms)
# and wild guesses

def make_features(df):
	# squared sum of horizontal distances is added for all horizontal distance types (hydrology, roadways, fire_points) and elevation
	# as well as the absolute difference between those distances
	df['Horz_Hydro_road_Sum_Sqr'] = (df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways']).pow(2,axis=0)
	df['Horz_Hydro_road_Sum_Sqr_dist'] = (df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways']).abs()

	df['Horz_Hydro_fire_Sum_Sqr'] = (df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']).pow(2,axis=0)
	df['Horz_Hydro_fire_Sum_Sqr_dist'] = (df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points']).abs()
	
	df['Horz_Hydro_elev_Sum_Sqr'] = (df['Horizontal_Distance_To_Hydrology'] + df['Elevation']).pow(2,axis=0)
	df['Horz_Hydro_elev_Sum_Sqr_dist'] = (df['Horizontal_Distance_To_Hydrology'] - df['Elevation']).abs()
	
	df['Vert_Hydro_elev_Sum_Sqr'] = (df['Vertical_Distance_To_Hydrology'] + df['Elevation']).pow(2,axis=0)
	df['Vert_Hydro_elev_Sum_Sqr_dist'] = (df['Vertical_Distance_To_Hydrology'] - df['Elevation']).abs()
	
	df['Horz_fire_elev_Sum_Sqr'] = (df['Horizontal_Distance_To_Fire_Points'] + df['Elevation']).pow(2,axis=0)
	df['Horz_fire_elev_Sum_Sqr_dist'] = (df['Horizontal_Distance_To_Fire_Points'] - df['Elevation']).abs()
	
	df['Horz_road_elev_Sum_Sqr'] = (df['Horizontal_Distance_To_Roadways'] + df['Elevation']).pow(2,axis=0)
	df['Horz_road_elev_Sum_Sqr_dist'] = (df['Horizontal_Distance_To_Roadways'] - df['Elevation']).abs()
	
	df['Horz_fire_road_Sum_Sqr'] = (df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points']).pow(2,axis=0)
	df['Horz_fire_road_Sum_Sqr_dist'] = (df['Horizontal_Distance_To_Roadways'] - df['Horizontal_Distance_To_Fire_Points']).abs()
	
	# wild guessing for aspect angle transformations
	df['Aspect_sin'] = (df['Aspect']).apply(np.sin)
	df['Aspect_cos'] = (df['Aspect']).apply(np.cos)
	df['Aspect_tan'] = (df['Aspect']).apply(np.tan)

	# some forests are under or below the hydrology(minus sign in the vertical distance)
	# which might be of use for our classifier to learn certain distinctions
	df['Hydro_below'] = df['Vertical_Distance_To_Hydrology'] < 0 

	# while having both horizontal and vertical distances, i assumed that the "real"
	# distance would be of use for the classifier(and it's square, why not), Pythagoras theorem
	df['Distance_to_Hydrology'] = (df['Horizontal_Distance_To_Hydrology'].pow(2,axis=0)
	                                + df['Vertical_Distance_To_Hydrology'].pow(2,axis=0)
	                                ).apply(np.sqrt)

	df['Distance_to_Hydrology_Sqr'] = df['Distance_to_Hydrology'].pow(2,axis=0)

	return df