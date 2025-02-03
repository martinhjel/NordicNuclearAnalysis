# 3. Cost parameters
#    -> update objective function
self.day = (timestep // 24) + 1  # Calculate the cumulative day count (starting at 1)
self.year = ((self.day - 1) // 365) + 1  # Determine the year (assuming 365 days per year)
day_of_year = ((self.day - 1) % 365) + 1  # Determine the day of the current year (1 through 365)
self.week = ((day_of_year - 1) // 7) + 1  # Calculate the week of the current year
if self.week > 52:  # Map self.week to a storage profile week between 1 and 52
    profile_week = ((self.week - 1) % 52) + 1
else:
    profile_week = self.week

# 3a. generators with storage (storage value)
for i in self._idx_generatorsWithStorage:
    this_type_filling = self._grid.generator.loc[i, "storval_filling_ref"]
    this_type_time = self._grid.generator.loc[i, "storval_time_ref"]
    storagecapacity = self._grid.generator.loc[i, "storage_cap"]
    fillinglevel = self._storage[i] / storagecapacity
    filling_col = int(round(fillinglevel * 100))
    if this_type_filling == 'hydro':
        week = f"week {profile_week}"
        storagevalue = (
                self._grid.generator.loc[i, "storage_price"]
                * self._grid.storagevalue_filling.loc[filling_col, week]
                * self._grid.storagevalue_time.loc[timestep, this_type_time]
        )
    else:
        storagevalue = (
                self._grid.generator.loc[i, "storage_price"]
                * self._grid.storagevalue_filling.loc[filling_col, this_type_filling]
                * self._grid.storagevalue_time.loc[timestep, this_type_time]
        )

    self.p_gen_cost[i] = storagevalue
    if i in self._idx_generatorsWithPumping:
        deadband = self._grid.generator.pump_deadband[i]
        self.p_genpump_cost[i] = storagevalue - deadband
