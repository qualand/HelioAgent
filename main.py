from copylot_to_soltrace import *
from concurrent.futures import ThreadPoolExecutor

NRAYS = 1.e6





def individual_heliostat_results(field: Heliostat_Field, PT: PySolTrace):
    # NOTE: This can take a while with many heliostats
    ST_results = {'power_to_receiver': list(),
                  'cosine_eff': list(),
                  'reflect_eff': list(),
                  'intercept_eff': list(),
                  'block_eff': list(),
                  'absorption_eff': list(),
                  'helio_total_eff': list(),
                  'total_eff': list()}
    
    def _calc_heliostat(args):
        i, h = args
        power = h.calculate_receiver_power(PT)
        efficiencies = h.calculate_efficiencies(PT)
        return i, power, efficiencies

    with ThreadPoolExecutor() as executor:
        helio_results = list(executor.map(_calc_heliostat, enumerate(field.heliostats)))

    for i, power, efficiencies in helio_results:
        ST_results['power_to_receiver'].append(power)
        ST_results['cosine_eff'].append(efficiencies['cosine'])
        ST_results['reflect_eff'].append(efficiencies['reflect'])
        ST_results['intercept_eff'].append(efficiencies['intercept'])
        ST_results['block_eff'].append(efficiencies['blocking'])
        ST_results['absorption_eff'].append(efficiencies['absorption'])
        ST_results['helio_total_eff'].append(efficiencies['helio_total'])
        ST_results['total_eff'].append(efficiencies['total'])

    ST_df = pd.DataFrame(ST_results)
    return ST_df

if __name__ == "__main__":
    run_sim_in_parallel = True
    results_dir = 'Results\\'

    # Creating initial solar field
    field = Heliostat_Field()
    field.rec_design_power = 12.0       # [MWt] Receiver design power
    field.tower_height = 50.0           # [m] Height of tower
    field.aim_method = 3                # Simple aim points=0; Sigma aiming=1; Probability shift=2; Image size priority=3; Keep existing=4; Freeze tracking=5
    field.include_attenuation = True    # Include atmospheric attenuation in SolTrace simulation
    
    # Simulation Parameters
    field.des_sim_ndays = 48 # 48, 24, 12, 4
    field.des_sim_nhours = 1

    # Field Layout Boundaries
    field.land_min_scaled_rad = 0.5
    field.land_max_scaled_rad = 20.0
    field.field_accept_max = 75.0
    field.field_accept_min = -75.0

    # Receiver Characteristics
    field.rec_type = 2                  # Flat plate
    field.rec_height =  4.0             # [m] Receiver height
    field.rec_width =  4.0              # [m] Receiver width
    field.rec_elevation = -30.0         # [deg] Receiver elevation angle

    # Heliostat Characteristics
    field.helio_height = 3.0            # [m] Heliostat height # TODO: These are different than the Ivanpah-like values
    field.helio_width = 4.55            # [m] Heliostat width
    field.helio_is_faceted = True       # Use multiple panels
    field.helio_n_cant_x = 2
    field.helio_n_cant_y = 1
    field.helio_surf_err = 0.002        # [rad] Heliostat surface error
    # TODO: update optical errors to be data driven

    # Field focusing
    field.helio_cant_method = 0         # No Canting
    field.helio_focus_method = 1                # 0 = Flat, 1 = At slant (each heliostat has a focal length dictated by its distance to the receiver)
    field.n_focus_bands = 3                     # helio_focus_method must be set to 1 
    field.focus_bands_method = 'average'        # 'mid-point', 'average', 'maximum'

    # Ivanpah-like Values:
    """
        self.helio_height = 4.167       # Heliostat height (m)
        self.helio_width = 6.085        # Heliostat width (m)
        self.helio_is_faceted = True    # Use multiple panels
        self.helio_n_cant_x = 2         # Number of canting facets in x (width) direction
        self.helio_n_cant_y = 1         # Number of canting facets in y (height) direction
        self.helio_x_gap = 0.085        # Gap between facets in x direction (m)
        self.helio_y_gap = 0.0          # Gap between facets in y direction (m)
    """

    field.generate_field_via_copylot(display_results = True)
    # plot_results(field.results['sp_detailed_field'], field.results['sp_incident_flux'], field.include_attenuation)   # Plot field, flux, and attenuation grouping
    PT = field.set_up_soltrace()        # Create SolTrace geometry
   
    #--- Update heliostat geometry based on solar position
    for h in field.heliostats:
        h.update_geometry(PT, field.results['sp_parameters']['fluxsim.0.flux_solar_az'], field.results['sp_parameters']['fluxsim.0.flux_solar_el'])
    # PT.write_soltrace_input_file('test_field_construction_check.stinput')

    # df, ppr, nhit = simulate_soltrace(PT, dni = 950., nray = NRAYS , seed = 123, nthreads = 12)
    # results = process_soltrace_results(PT, field.field_area)
    # print("Receiver absorbed power: {:.2f} (kW)".format(results['Absorbed power (kW)']))

    # ST_df = individual_heliostat_results(field, PT)
    # print("Total power to receiver from heliostat calculations: {:.2f} (kW)".format(ST_df['power_to_receiver'].sum()))

    if False:
        weather_data = pd.read_csv(field.weather_file, skiprows=2)
        resource_data = weather_data[weather_data['DNI'] > 10.].reset_index()

        resource_data['Tdry'].hist(density=True, bins=15, label='Resource Data')
        field.results['sp_sim_data']['temp'].hist(density=True, bins=15, label='Simulated Data', alpha=0.7)
        plt.ylabel("Probability Density")
        plt.xlabel("Temperature [C]")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(results_dir + 'temperature_distribution_comparison.png', dpi=300)

        resource_data['DNI'].hist(density=True, bins=15, label='Resource Data')
        field.results['sp_sim_data']['dni'].hist(density=True, bins=15, label='Simulated Data', alpha=0.7)
        plt.ylabel("Probability Density")
        plt.xlabel("DNI [W/m^2]")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(results_dir + 'dni_distribution_comparison.png', dpi=300)


    # Time-series simulation
    print("Number of simulation time steps: {:d}".format(len(field.results['sp_sim_data'])))
    tstart = time.time()
    # Parallelization:
    if run_sim_in_parallel:
        with Pool() as pool:
            baseline_results_list = pool.starmap(simulate_time_step, [(PT, field, row, NRAYS) for _, row in field.results['sp_sim_data'].iterrows()])
            correction_results_list = pool.starmap(simulate_time_step_with_FL_correction, [(PT, field, row, NRAYS) for _, row in field.results['sp_sim_data'].iterrows()])
    else: # Serial execution
        baseline_results_list = []
        correction_results_list = []
        for idx, row in field.results['sp_sim_data'].iterrows():
            baseline_results_list.append(simulate_time_step(PT, field, row, NRAYS, nthreads=14))
            correction_results_list.append(simulate_time_step_with_FL_correction(PT, field, row, NRAYS, nthreads=14))

    print("\nSimulation complete. Total simulation time {:.2f} seconds.".format(time.time()-tstart))

    sim_data_copy = field.results['sp_sim_data'].copy() # Create a copy of the original dataframe to store results
    baseline_sim_results = pd.concat([sim_data_copy, pd.DataFrame(baseline_results_list)], axis=1)
    correction_sim_results = pd.concat([sim_data_copy, pd.DataFrame(correction_results_list)], axis=1)

    # Save results to CSV
    baseline_sim_results.to_csv(results_dir + 'baseline_sim_results.csv', index=False)
    correction_sim_results.to_csv(results_dir + 'correction_sim_results.csv', index=False)

    # Report results
    baseline_absorbed_energy_sum = sum(baseline_sim_results['Absorbed power (kW)'])
    correction_absorbed_energy_sum = sum(correction_sim_results['Absorbed power (kW)'])

    print("sum of absorbed energy (baseline): {:,.2f} kWh".format(baseline_absorbed_energy_sum))
    print("sum of absorbed energy (correction): {:,.2f} kWh".format(correction_absorbed_energy_sum))
    print("Difference in absorbed energy: {:,.2f} kWh".format(correction_absorbed_energy_sum - baseline_absorbed_energy_sum))
    print("Percentage difference in absorbed energy: {:.2f} %".format(100. * (correction_absorbed_energy_sum - baseline_absorbed_energy_sum) / baseline_absorbed_energy_sum))

    # Plot optical efficiency and temperature over time
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(baseline_sim_results.index, baseline_sim_results['Field efficiency (%)'], 'b-', linewidth=2, label='baseline')
    ax1.plot(correction_sim_results.index, correction_sim_results['Field efficiency (%)'], 'b--', linewidth=2, label='correction')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Optical Efficiency [%]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(baseline_sim_results.index, baseline_sim_results['temp'], 'r-', linewidth=2, label='Temperature')
    ax2.set_ylabel('Temperature [°C]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    fig.suptitle('Optical Efficiency and Temperature Over Time')
    fig.tight_layout()
    # plt.show()
    plt.savefig(results_dir + 'efficiency_temperature_time_series.png', dpi=300)

    # Plot difference in field efficiency vs. temperature
    field_efficiency_diff = correction_sim_results['Field efficiency (%)'] - baseline_sim_results['Field efficiency (%)']
    plt.figure()
    plt.scatter(baseline_sim_results['temp'], field_efficiency_diff, alpha=0.7)
    plt.xlabel('Temperature [°C]')
    plt.ylabel('Difference in Field Efficiency [%]')
    plt.title('Difference in Field Efficiency vs. Temperature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(results_dir + 'efficiency_difference_vs_temperature.png', dpi=300)

        # TODO:
        # - Initialize field with focal length bands
        #      - Field size, receiver capacity, etc.
        # - Set up focal length update based on elevation angle (gravity), temperature, and wind speed / direction
        #      - Gravity (waiting on data)
        #      - Temperature (have dummy data)
        #      - Wind speed / direction (waiting on data)
        # - Create connection with SAM to get efficiency table
