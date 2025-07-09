from altair import to_csv
import pandas as pd


# Load the dataset correctly with the correct delimiter
mat_df = pd.read_csv('datasets/student-mat.csv', delimiter=';')

# Check the shape of the DataFrame
#print(mat_df.shape)

# Save the DataFrame back to CSV
#mat_df.to_csv('student_mat.csv', index=False)

por_df=pd.read_csv('datasets/student-por.csv', delimiter=';')

# Check the shape of the DataFrame
print(por_df.shape)

# Save the DataFrame back to CSV
por_df.to_csv('student_por.csv', index=False)

