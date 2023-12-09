# class Collaborator:

#     def __init__(self) -> None:

#         self.age = age
#         self.income = income
#         self.gender = gender
#         self.height = height
#         self.weight = weight
#         self.family_size = family_size
#         self.education = education
#         self.marital_status = marital_status
#         self.occupation = occupation
#         self.housing = housing
#         self.region = region
#         self.sub_region = sub_region
#         self.country_of_origin = country_of_origin
#         self.ethnicity = ethnicity

#         pass

import matplotlib.pyplot as plt
import numpy as np

# Generate data
drone_distances = list(np.arange(1, 11, 0.1))
stress_levels = [10 / (distance ** 2) for distance in drone_distances]

# Plotting
plt.plot(drone_distances, stress_levels, marker='')

# Adding labels and title
plt.xlabel('Drone Distance (m)')
plt.ylabel('Stress')
plt.title('Stress vs Drone Distance')

# Display the plot
plt.grid(True)
plt.show()
