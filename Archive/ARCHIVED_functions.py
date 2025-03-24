# ARCHIVED FUNCTIONS THAT ARE NO LONGER IN USE


def get_time_steps_for_years(selected_years):
    """
    Given a list of specific years, return a dictionary with min and max time step indices for each year.

    Parameters:
    selected_years (list of int): The years to include (between 1991 and 2020).

    list_of_years = get_time_steps_for_years(selected_years=[1993, 2001, 2009, 2018])

    Returns:
    dict: {year: [min_time_step, max_time_step]} for each selected year.
    """
    first_year = 1991
    last_year = 2020
    leap_years = {1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020}

    # Validate input years
    if any(year < first_year or year > last_year for year in selected_years):
        raise ValueError(f"Years must be between {first_year} and {last_year}")

    # Sort the years to process them in order
    selected_years = sorted(selected_years)

    # Initialize tracking of time steps
    min_time_step = 0
    year_time_steps = {}

    # Loop through all years to calculate min/max for each selected year
    for year in range(first_year, last_year + 1):
        year_hours = 8784 if year in leap_years else 8760  # Handle leap years

        if year in selected_years:
            max_time_step = min_time_step + year_hours - 1
            year_time_steps[year] = [min_time_step, max_time_step]

        # Move to next year
        min_time_step += year_hours

    return year_time_steps


def get_time_steps_for_period(start_year, end_year):
    """
    Given a start and end year, return the corresponding min and max time step indices.

    Parameters:
    start_year (int): The starting year of the period (between 1991 and 2020).
    end_year (int): The ending year of the period (between 1991 and 2020).

    # Example usage:
    timeMaxMin_YEAR = get_time_steps_for_period(2000, 2000)

    Returns:
    list: [min_time_step, max_time_step]
    """
    # Define the valid range
    first_year = 1991
    last_year = 2020

    # Leap years that have 8,784 hours instead of 8,760
    leap_years = {1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020}

    # Validate input years
    if start_year < first_year or end_year > last_year or start_year > end_year:
        raise ValueError(f"Years must be between {first_year} and {last_year}, with start_year ≤ end_year")

    # Compute min time step (start of start_year)
    min_time_step = 0
    for y in range(first_year, start_year):
        min_time_step += 8784 if y in leap_years else 8760

    # Compute max time step (end of end_year)
    max_time_step = min_time_step  # Start at the min time step
    for y in range(start_year, end_year + 1):  # Include the last year
        max_time_step += 8784 if y in leap_years else 8760

    return [min_time_step, max_time_step - 1]  # -1 to get the last valid index




