import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

def set_to_floats(pop_time):
    pop_time['maxeff_arm_2b_3b_sba'] = pop_time['maxeff_arm_2b_3b_sba'].astype(float)
    pop_time['exchange_2b_3b_sba'] = pop_time['exchange_2b_3b_sba'].astype(float)
    pop_time['pop_3b_sba_count'] = pop_time['pop_3b_sba_count'].astype(float)
    pop_time['pop_3b_sba'] = pop_time['pop_3b_sba'].astype(float)
    pop_time['pop_3b_cs'] = pop_time['pop_3b_cs'].astype(float)
    pop_time['pop_3b_sb'] = pop_time['pop_3b_sb'].astype(float)
    pop_time['pop_2b_sba'] = pop_time['pop_2b_sba'].astype(float)
    pop_time['pop_2b_cs'] = pop_time['pop_2b_cs'].astype(float)
    pop_time['pop_2b_sb'] = pop_time['pop_2b_sb'].astype(float)
    pop_time["Percentile"] = np.nan
    pop_time['Percentile'] = pop_time['Percentile'].astype(float)

def set_to_floats_2(table):
    table['lead_distance_gained'] = table['lead_distance_gained'].replace('--', np.nan).astype(float)
    table['at_pitchers_first_move'] = table['at_pitchers_first_move'].replace('--', np.nan).astype(float)
    table['at_pitch_release'] = table['at_pitch_release'].replace('--', np.nan).astype(float)
    table['lead_distance_gained'] = table['lead_distance_gained'].astype(float)
    table['at_pitchers_first_move'] = table['at_pitchers_first_move'].astype(float)
    table['at_pitch_release'] = table['at_pitch_release'].astype(float)

def set_to_floats_3(running_splits):
    running_splits['seconds_since_hit_000'] = running_splits['seconds_since_hit_000'].astype(float)
    running_splits['seconds_since_hit_005'] = running_splits['seconds_since_hit_005'].astype(float)
    running_splits['seconds_since_hit_010'] = running_splits['seconds_since_hit_010'].astype(float)
    running_splits['seconds_since_hit_015'] = running_splits['seconds_since_hit_015'].astype(float)
    running_splits['seconds_since_hit_020'] = running_splits['seconds_since_hit_020'].astype(float)
    running_splits['seconds_since_hit_025'] = running_splits['seconds_since_hit_025'].astype(float)
    running_splits['seconds_since_hit_030'] = running_splits['seconds_since_hit_030'].astype(float)
    running_splits['seconds_since_hit_035'] = running_splits['seconds_since_hit_035'].astype(float)
    running_splits['seconds_since_hit_040'] = running_splits['seconds_since_hit_040'].astype(float)
    running_splits['seconds_since_hit_045'] = running_splits['seconds_since_hit_045'].astype(float)
    running_splits['seconds_since_hit_050'] = running_splits['seconds_since_hit_050'].astype(float)
    running_splits['seconds_since_hit_055'] = running_splits['seconds_since_hit_055'].astype(float)
    running_splits['seconds_since_hit_060'] = running_splits['seconds_since_hit_060'].astype(float)
    running_splits['seconds_since_hit_065'] = running_splits['seconds_since_hit_065'].astype(float)
    running_splits['seconds_since_hit_070'] = running_splits['seconds_since_hit_070'].astype(float)
    running_splits['seconds_since_hit_075'] = running_splits['seconds_since_hit_075'].astype(float)
    running_splits['seconds_since_hit_080'] = running_splits['seconds_since_hit_080'].astype(float)
    running_splits['seconds_since_hit_085'] = running_splits['seconds_since_hit_085'].astype(float)
    running_splits['seconds_since_hit_090'] = running_splits['seconds_since_hit_090'].astype(float)


'''     Turns full name e.g. Shohei Ohtani into Ohtani, S.      '''
def shorten_name(name):
    # Split the name into last name and first name
    last_name, first_name = name.split(', ')
    
    # Get the first letter of the first name
    first_initial = first_name[0] + '.'
    
    # Combine last name and first initial
    shortened_name = f"{last_name}, {first_initial}"
    
    return shortened_name

def getVelo():
    while True:
        velo = input("Give me the velocity of the pitch(mph): ")
        try:
            # Try to convert the input to an float
            velo = float(velo)
            break  # Break the loop if conversion is successful (i.e., input is valid)
        except ValueError:
            print("That's not a valid velo. Please try again.")
    return velo

def getCatcher(catcher):

    catcher_poptime = 0.0
    while catcher_poptime == 0.0:
        find_catcher = pop_time.loc[pop_time['catcher'] == catcher, 'pop_2b_sba']
        if not find_catcher.empty:
            catcher_poptime = find_catcher.values[0]
        else:
            print("No catcher data found.")
            catcher = input("Give me another catcher(Put First initial. Last name e.g. J. Realmuto) \n ")
    return catcher_poptime


def getJump():
    while True:
        jump = input("Enter a jump between 0 and 10: ")
        
        try:
            # Convert the input to a float or int
            jump = int(jump)
            
            # Check if the number is within the desired range
            if 0 <= jump <= 10:
                break  # Valid input, exit the loop
            else:
                print("The number is out of range. Please try again.")
        
        except ValueError:
            # Handle the case where input is not a valid number
            print("Invalid input. Please enter a valid number.")
    return jump

def reformat_name(name):
    # Split by comma and strip extra spaces
    last, first_initial = name.split(', ')
    # Return the name in the format 'First Initial. Last'
    return f"{first_initial} {last}"

'''         Read all running data            '''
table = pd.read_csv("./fulltable.csv")
set_to_floats_2(table)
table = table.drop(columns=['Catcher', 'Fielder', 'pitcher_stealing_runs', 'lead_distance_gained'])
table = table.sort_values(by = "at_pitch_release")
table = table.sort_values(by = "Outcome")

'''         Print table with Catcher's pop time's        '''
pop_time = pd.read_csv("./poptime.csv")
set_to_floats(pop_time)
pop_time = pop_time.drop(columns=['player_id','age','maxeff_arm_2b_3b_sba', 'exchange_2b_3b_sba', 'pop_3b_sba_count', 'pop_3b_sba', 'pop_3b_cs', 'pop_3b_sb' ])
pop_time = pop_time.sort_values(by = "pop_2b_sba", ascending = [True])
pop_time.reset_index(drop = True, inplace=True)
total_rows = len(pop_time)

for i, row in pop_time.iterrows():
    name = row['catcher']
    shortened_name = shorten_name(name)
    pop_time.loc[i, 'catcher'] = shortened_name


table['Runner'] = table['Runner'].apply(reformat_name)
pop_time['catcher'] = pop_time['catcher'].apply(reformat_name)

'''         Ask user input             '''
velo = getVelo()
catcher = input("Give me the name of the catcher(Put First initial. Last name e.g. J. Realmuto) \n")
catcher_poptime = getCatcher(catcher)
runner = input("Give me a baserunner - (First initial.  Last name e.g. S. Ohtani) \n")


#Find pitch velocity 
pitch_time = ((60.6/velo)/5280)* 3600
fielding_time = pitch_time + catcher_poptime

#Create baserunner table
table2 = pd.DataFrame()

table2 = pd.DataFrame()
while table2.empty:
    table2 = table.copy()
    for i, row in table2.iterrows():
        temp = row['Runner']
        if temp != runner:
            table2.drop(i, axis = 0, inplace = True)
    if table2.empty:
        print("No baserunner data found.")
        runner = input("Give me another baserunner - (First initial.  Last name e.g. S. Ohtani)\n ")

jump = getJump()
#Find the distance using the jump input
min_jump = table2['at_pitch_release'].min()
max_jump = table2['at_pitch_release'].max()
min_jump =  min_jump.astype(float)
max_jump = max_jump.astype(float)

# print(min_jump, max_jump)
jump_range = max_jump - min_jump

new_jump = (jump_range * (jump / 10)) + min_jump
float_jump = float(jump)

#Grab Sprint Speed 
remaining_distance = 60.0 - new_jump
running_splits = pd.read_csv("./running_splits.csv")
set_to_floats_3(running_splits)
running_splits = running_splits.drop(columns=['player_id','team_id', 'name_abbrev', 'position_name', 'age'])
running_splits['last_name, first_name'] = running_splits['last_name, first_name'].apply(reformat_name)
runner_time = 0.0

line_graph = pd.DataFrame(columns=['Distance', 'Sprint Speed'])
i = 0
j = 2
while i < 95:
    s_speed = running_splits.iloc[0, j]
    line_graph.loc[len(line_graph)] = [i, s_speed]  # Adding a row at the next index
    i += 5
    j += 1

# Define the exponential function
def power_func(x, a, b):
    return a * np.power(x, b)


# Fit the exponential model to the data
params, covariance = curve_fit(power_func, line_graph['Distance'], line_graph['Sprint Speed'])
# Extract the fitted parameters
a, b = params
# Generate data for plotting the fitted curve
x_fit = np.linspace(min(line_graph['Distance']), max(line_graph['Distance']), 100)
y_fit = power_func(x_fit, a, b)
# Plot the original data and the fitted curve
plt.scatter(line_graph['Distance'], line_graph['Sprint Speed'], color='black', label='Data points')
plt.plot(x_fit, y_fit, color='red', label='Exponential fit')
plt.xlabel('Distance')
plt.ylabel('Sprint Speed')
plt.title('Exponential Fit')
plt.legend()
plt.grid(True)
intercept = power_func(1, a, b)
equation = f"y = {a:.2f} * x^{b:.2f}"
result = 0.0
result = 0.17 * (remaining_distance ** 0.70) + 0.16850455562198682

copy = pd.DataFrame(columns=['Distance', 'Sprint Speed'])
w = 0
new_speed = 0.0
while w < 95:
    new_speed = 0.17 * (w ** 0.70) + 0.16850455562198682
    copy.loc[len(copy)] = [w, new_speed]  # Adding a row at the next index
    w += 5
    
plt.show()

print(f"With a pitch velo of {velo} and a jump of {jump}")
print(f"The remaining distance that {runner} needs to cover is {remaining_distance}")
print(f"The pitch time ends up being {pitch_time} and {catcher}'s poptime is {catcher_poptime} giving a total fielding time of {fielding_time}")
print(f"Factoring sprint speed with the distance remaining the time it takes for {runner} to steal 2nd is {result}")
print("\n")

if(result < fielding_time):
    print("The runner gets there first, HES SAFE!!!")
elif (result > fielding_time):
    print("The ball gets there first, HES OUT!!!")
else:
    print("They have the same time!")

