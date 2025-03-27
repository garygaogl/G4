
# load necessary libraries
library(ggplot2)    # Visualization
library(dplyr)      # Data manipulation
library(cluster)    # Distance calculation for clustering
library(factoextra) # Enhanced visualization of clustering results


# Read the data
df <- read.csv("/Users/chukexu/Downloads/Dataset_ATS_v2-checkpoint.csv", stringsAsFactors = FALSE)

# Check the data structure
str(df)

# Select numerical variables
df_cluster <- df %>% select(tenure, MonthlyCharges)

# Scale the data
# Since K-Means is sensitive to different data ranges,
# standardizing the features using scale() to give them equal weight 
# in distance calculations
df_cluster <- scale(df_cluster)

# Check scaled data
head(df_cluster)

# Compute WCSS using Elbow Method
set.seed(42)  
fviz_nbclust(df_cluster, kmeans, method = "wss") +
  ggtitle("Elbow Method to Find Optimal K")
# Insight: Helps choose an appropriate number of clusters 
# to avoid underfitting or overfitting.

# Train the K-Means model
set.seed(42)
kmeans_model <- kmeans(df_cluster, centers = 3, nstart = 25)
# Insight: From the image, we can see there is a turning point at 3.


# View the size of each cluster
kmeans_model$size
fviz_cluster(kmeans_model, data = df_cluster, geom = "point", ellipse.type = "norm") +
  ggtitle("Customer Clusters using K-Means")
df$Cluster <- as.factor(kmeans_model$cluster)

# View the first few rows of updated dataset
head(df)

