from copylot_to_soltrace import *
from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":
    # Creating initial solar field
    field = Heliostat_Field()
    field.rec_design_power = 50.       # [MWt] Receiver design power
    field.aim_method = 3               # Simple aim points=0; Sigma aiming=1; Probability shift=2; Image size priority=3; Keep existing=4; Freeze tracking=5
    field.include_attenuation = True   # Include atmospheric attenuation in SolTrace simulation

    # Heliostat parameters
    field.helio_focus_method = 1        # 0 = Flat, 1 = At slant (each heliostat has a focal length dictated by its distance to the receiver)
    
    field.generate_field_via_copylot(display_results = True)
    #plot_results(field.results['sp_detailed_field'], field.results['sp_incident_flux'], field.include_attenuation)   # Plot field, flux, and attenuation grouping
    PT = field.set_up_soltrace()        # Create SolTrace geometry
   
    #--- Update heliostat geometry based on solar position
    for h in field.heliostats:
        h.update_geometry(PT, field.results['sp_parameters']['fluxsim.0.flux_solar_az'], field.results['sp_parameters']['fluxsim.0.flux_solar_el'])
    PT.write_soltrace_input_file('test_field_construction.stinput')

    df, ppr, nhit = simulate_soltrace(PT, dni = 950., nray = 10.e6 , seed = 123, nthreads = 12)
    results = process_soltrace_results(PT, field.field_area)
    print("Receiver absorbed power: {:.2f} (kW)".format(results['Absorbed power (kW)']))


    # Change focal lengths
    for h in field.heliostats:
        h.update_focal_length(-500.0)   # Set all heliostat focal lengths to -500 m
    PT.write_soltrace_input_file('update_focal_length_negative.stinput')

    df, ppr, nhit = simulate_soltrace(PT, dni = 950., nray = 10.e6 , seed = 123, nthreads = 12)
    results = process_soltrace_results(PT, field.field_area)
    print("Receiver absorbed power (negative FL): {:.2f} (kW)".format(results['Absorbed power (kW)']))

    for h in field.heliostats:
        h.update_geometry(PT, 135.0, 30.0)   # Update geometry for new solar position
    PT.write_soltrace_input_file('update_focal_length_negative_wUpdate.stinput')

    df, ppr, nhit = simulate_soltrace(PT, dni = 950., nray = 10.e6 , seed = 123, nthreads = 12)
    results = process_soltrace_results(PT, field.field_area)
    print("Receiver absorbed power (negative FL with Update): {:.2f} (kW)".format(results['Absorbed power (kW)']))


    for h in field.heliostats:
        slant_focal_length = (h.position.x**2
                              + h.position.y**2
                              + field.tower_height**2 )**0.5
        h.update_focal_length(slant_focal_length)
        h.update_geometry(PT, field.results['sp_parameters']['fluxsim.0.flux_solar_az'], field.results['sp_parameters']['fluxsim.0.flux_solar_el'])

    PT.write_soltrace_input_file('test_field_back_to_original.stinput')

    df, ppr, nhit = simulate_soltrace(PT, dni = 950., nray = 10.e6 , seed = 123, nthreads = 12)
    results = process_soltrace_results(PT, field.field_area)
    print("Receiver absorbed power (back to original): {:.2f} (kW)".format(results['Absorbed power (kW)']))
