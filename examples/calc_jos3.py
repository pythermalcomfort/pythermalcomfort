import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pythermalcomfort.jos3_functions.parameters import local_clo_typical_ensembles
from pythermalcomfort.models import JOS3

# Make "jos3_example_example" directory in the current directory
directory_name = "jos3_output_example"
current_directory = os.getcwd()
jos3_example_directory = os.path.join(current_directory, directory_name)
if not os.path.exists(jos3_example_directory):
    os.makedirs(jos3_example_directory)

# -------------------------------------------
# EXAMPLE 1 (simple simulation)
# -------------------------------------------


def simple_simulation():
    """Run a simple simulation example."""
    # Build a model and set a body built
    # Create an instance of the JOS3 class with optional body parameters such as body height, weight, age, sex, etc.
    model = JOS3(height=1.7, weight=60, age=30)

    # Set the first phase
    model.to = 28  # Operative temperature [°C]
    model.rh = 40  # Relative humidity [%]
    model.v = 0.2  # Air velocity [m/s]
    model.par = 1.2  # Physical activity ratio [-]
    model.simulate(60)  # Exposure time = 60 [min]

    # Set the next condition (You only need to change the parameters that you want to change)
    model.to = 20  # Change only operative temperature
    model.simulate(60)  # Additional exposure time = 60 [min]

    # Show the results
    df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
    df.t_skin_mean.plot()  # Plot time series of mean skin temperature.
    plt.ylabel(
        "Mean skin temperature [°C]",
    )  # Set y-label as 'Mean skin temperature [°C]'
    plt.xlabel("Time [min]")  # Set x-label as 'Time [min]'
    plt.savefig(
        os.path.join(jos3_example_directory, "jos3_example1_mean_skin_temperature.png"),
    )  # Save plot at the current directory
    plt.show()  # Show the plot

    # Exporting the results as csv
    model.to_csv(
        os.path.join(jos3_example_directory, "jos3_example1 (default output).csv"),
    )

    # Print the BMR value using the getter
    print("BMR=", model.bmr)
    print("Body name list: ", model.body_names)


def detailed_simulation():
    """Run a detailed simulation example with multiple phases."""
    # Build a model and set a body built
    # Create an instance of the JOS3 class with optional body parameters such as body height, weight, age, sex, etc.
    model = JOS3(
        height=1.7,
        weight=60,
        fat=20,
        age=30,
        sex="male",
        bmr_equation="japanese",
        bsa_equation="fujimoto",
    )

    # Set environmental conditions such as air temperature, mean radiant temperature using the setter methods.
    # Set the first condition
    # Environmental parameters can be input as int, float, list, dict, numpy array format.
    model.tdb = 28  # Air temperature [°C]
    model.tr = 30  # Mean radiant temperature [°C]
    model.rh = 40  # Relative humidity [%]
    model.v = np.asarray(  # Air velocity [m/s]
        [
            0.2,  # head
            0.4,  # neck
            0.4,  # chest
            0.1,  # back
            0.1,  # pelvis
            0.4,  # left shoulder
            0.4,  # left arm
            0.4,  # left hand
            0.4,  # right shoulder
            0.4,  # right arm
            0.4,  # right hand
            0.1,  # left thigh
            0.1,  # left leg
            0.1,  # left foot
            0.1,  # right thigh
            0.1,  # right leg
            0.1,  # right foot
        ],
    )
    model.clo = local_clo_typical_ensembles[
        "briefs, socks, undershirt, work jacket, work pants, safety shoes"
    ]["local_body_part"]

    # par should be input as int, float.
    model.par = 1.2  # Physical activity ratio [-], assuming a sitting position
    # posture should be input as int (0, 1, or 2) or str ("standing", "sitting" or "lying").
    # (0="standing", 1="sitting" or 2="lying")
    model.posture = "sitting"  # Posture [-], assuming a sitting position

    # Run JOS-3 model
    model.simulate(
        times=30,  # Number of loops of a simulation
        dtime=60,  # Time delta [sec]. The default is 60.
    )  # Exposure time = 30 [loops] * 60 [sec] = 30 [min]

    # Set the next condition (You only need to change the parameters that you want to change)
    model.to = 20  # Change operative temperature
    model.v = {  # Air velocity [m/s], assuming to use a desk fan
        "head": 0.2,
        "neck": 0.4,
        "chest": 0.4,
        "back": 0.1,
        "pelvis": 0.1,
        "left_shoulder": 0.4,
        "left_arm": 0.4,
        "left_hand": 0.4,
        "right_shoulder": 0.4,
        "right_arm": 0.4,
        "right_hand": 0.4,
        "left_thigh": 0.1,
        "left_leg": 0.1,
        "left_foot": 0.1,
        "right_thigh": 0.1,
        "right_leg": 0.1,
        "right_foot": 0.1,
    }
    # Run JOS-3 model
    model.simulate(
        times=60,  # Number of loops of a simulation
        dtime=60,  # Time delta [sec]. The default is 60.
    )  # Additional exposure time = 60 [loops] * 60 [sec] = 60 [min]

    # Set the next condition (You only need to change the parameters that you want to change)
    model.tdb = 30  # Change air temperature [°C]
    model.tr = 35  # Change mean radiant temperature [°C]
    # Run JOS-3 model
    model.simulate(
        times=30,  # Number of loops of a simulation
        dtime=60,  # Time delta [sec]. The default is 60.
    )  # Additional exposure time = 30 [loops] * 60 [sec] = 30 [min]

    # Show the results
    df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
    df[
        ["t_skin_mean", "t_skin_head", "t_skin_chest", "t_skin_left_hand"]
    ].plot()  # Plot time series of local skin temperature.
    plt.legend(["Mean", "Head", "Chest", "Left hand"])  # Reset the legends
    plt.ylabel("Skin temperature [°C]")  # Set y-label as 'Skin temperature [°C]'
    plt.xlabel("Time [min]")  # Set x-label as 'Time [min]'
    plt.savefig(
        os.path.join(jos3_example_directory, "jos3_example2_skin_temperatures.png"),
    )  # Save plot at the current directory
    plt.show()  # Show the plot

    # Exporting the results as csv
    model.to_csv(os.path.join(jos3_example_directory, "jos3_example2 (all output).csv"))


def female_simulation():
    """Run a simple simulation example."""
    model = JOS3(height=1.7, weight=60, age=30, sex="female", bmr_equation="japanese")

    # Set the first phase
    model.to = 28
    model.rh = 40
    model.v = 0.2
    model.par = 1.2
    model.clo = 0.3
    model.simulate(60)

    # Set the next condition
    model.to = 20
    model.simulate(60)

    # Show the results
    df = pd.DataFrame(model.dict_results())
    df.t_skin_mean.plot()
    plt.ylabel("Mean skin temperature [°C]")
    plt.xlabel("Time [min]")
    plt.savefig(
        os.path.join(jos3_example_directory, "jos3_example3_mean_skin_temperature.png"),
    )
    plt.show()

    # Exporting the results as csv
    model.to_csv(
        os.path.join(jos3_example_directory, "jos3_example3 (female simulation).csv"),
    )


def validation_simulation():
    """Run validation simulation using Stolwijk and Hardy dataset."""
    """
    Following code is for validation between experimental and predicted data
    """
    # Initialize an empty dictionary to hold the datasets
    exp_dataset = {}

    # Datasets are stored as CSV so that running the examples needs no Excel
    # reader. They are exported verbatim from
    # human_subject_experiment_dataset.xlsx, which is kept alongside them as the
    # archival source; see export_experiment_dataset_to_csv.py.
    dataset_names = ["Stolwijk1966", "Werner1980"]

    for dataset_name in dataset_names:
        dataset_path = os.path.join(
            jos3_example_directory,
            f"human_subject_experiment_dataset_{dataset_name}.csv",
        )
        try:
            # float_precision="round_trip" makes the parsed values bit-identical
            # to the original spreadsheet; the default parser can be 1 ULP off.
            exp_dataset[dataset_name] = pd.read_csv(
                dataset_path,
                header=0,
                float_precision="round_trip",
            )
        # Handle the case where the file is not found
        except FileNotFoundError:
            print(f"File {dataset_path} not found.")
        # Handle other general exceptions
        except Exception as e:
            print(f"An error occurred: {e}")

    # Concatenate all the individual data frames into a single DataFrame
    sim_dataset = {}

    def sim_stolwijk_hardy(models, tolist, rhlist):
        result = []
        for model in models:
            model.icl = np.asarray(
                [
                    0,
                    0,
                    0,
                    0.3,
                    0.3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.3,
                    0.3,
                    0,
                    0.3,
                    0.3,
                    0,
                ],
            )
            model.par = 1.2
            model.posture = "sitting"
            # model.options["nonshivering_thermogenesis"] = True
            # model.options["limit_dshiv/dt"] = True
            # model.options["shivering_threshold"] = True

            model.to = tolist[0]
            model.rh = rhlist[0]
            model.simulate(10, dtime=600)

            model.to = tolist[1]
            model.rh = rhlist[1]
            model.simulate(60)

            model.to = tolist[2]
            model.rh = rhlist[2]
            model.simulate(120)

            model.to = tolist[3]
            model.rh = rhlist[3]
            model.simulate(60)

            sim = pd.DataFrame(model.dict_results())
            sim = sim.loc[
                10:,
                "t_skin_mean":,
            ]
            result.append(sim.copy())

        avgsim = (result[0] + result[1] + result[2]) / 3
        avgsim = avgsim.iloc[::5]
        avgsim.index = [i for i in range(0, 241, 5)]
        avgsim["time"] = [i for i in range(0, 241, 5)]

        return avgsim.copy()

    simdatas = {}

    # -------------------------------------------------------------------------------
    # A : Stolwijk and Hardy, 1966
    # J.A.J. Stolwijk, J.D. Hardy, Partitional calorimetric studies of responses of man
    # to thermal transients, J. Appl. Physiol. 21 (3) (1966) 967–977,
    # https://doi.org/10.1152/jappl.1966.21.6.1799.
    # -------------------------------------------------------------------------------
    dnames = ["A-FIG.4", "A-FIG.5", "A-FIG.6", "A-FIG.7"]

    inputcons = {
        "A-FIG.4": {"to": [28.0, 27.8, 33.3, 28.0], "rh": [40, 37, 34, 37]},
        "A-FIG.5": {"to": [28.0, 28.5, 37.5, 28.5], "rh": [40, 40, 34, 41]},
        "A-FIG.6": {"to": [28.0, 28.0, 42.5, 28.1], "rh": [40, 37, 34, 37]},
        "A-FIG.7": {"to": [28.0, 28.1, 47.8, 28.3], "rh": [40, 43, 27, 44]},
    }

    for d in dnames:
        models = []
        models.append(
            JOS3(
                height=1.95,
                weight=88.6,
                age=25,
            ),
        )
        models.append(
            JOS3(
                height=1.84,
                weight=76.1,
                age=22,
            ),
        )
        models.append(
            JOS3(
                height=1.75,
                weight=110,
                age=23,
            ),
        )

        tolist = inputcons[d]["to"]
        rhlist = inputcons[d]["rh"]
        output = sim_stolwijk_hardy(models, tolist, rhlist)
        print("simulated:", d)

        simdatas[d] = output

    # -------------------------------------------------------------------------------
    # B : Stolwijk and Hardy, 1966
    #  J.D. Hardy, J.A.J. Stolwjk, Partitional exposures calorimetric studies of man
    # during exposures to thermal transients, J. Appl. Physiol. 21 (1966) 1799–1806,
    # https://doi.org/10.1152/jappl.1966.21.6.1799.
    # -------------------------------------------------------------------------------
    dnames = ["B-FIG.1", "B-FIG.2", "B-FIG.3", "B-FIG.4", "B-FIG.5"]

    inputcons = {
        "B-FIG.1": {"to": [28.0, 29.0, 22.0, 29.0], "rh": [40, 44, 39, 41]},
        "B-FIG.2": {"to": [28.0, 28.0, 18.0, 28.0], "rh": [40, 40, 40, 40]},
        "B-FIG.3": {"to": [28.0, 22.3, 43.5, 22.6], "rh": [40, 40, 38, 36]},
        "B-FIG.4": {"to": [28.0, 18.0, 42.0, 18.0], "rh": [40, 40, 40, 40]},
        "B-FIG.5": {"to": [28.0, 43.0, 17.0, 43.0], "rh": [40, 40, 40, 40]},
    }

    for d in dnames:
        models = []
        models.append(
            JOS3(
                height=1.91,
                weight=77.2,
                age=25,
            ),
        )
        models.append(
            JOS3(
                height=1.91,
                weight=84.5,
                age=26,
            ),
        )
        models.append(
            JOS3(
                height=1.88,
                weight=92.7,
                age=22,
            ),
        )

        tolist = inputcons[d]["to"]
        rhlist = inputcons[d]["rh"]
        output = sim_stolwijk_hardy(models, tolist, rhlist)
        print("simulated:", d)

        simdatas[d] = output

    to_concat = []
    for key, value in simdatas.items():
        value["Condition"] = key
        to_concat.append(value.copy())
    df = pd.concat(to_concat).reset_index(drop=True)

    sim_dataset["Stolwijk1966"] = df.copy()

    gexp = exp_dataset["Stolwijk1966"].copy()
    gsim = sim_dataset["Stolwijk1966"].copy()
    dfdict = {}
    for dn in [
        "A-FIG.4",
        "A-FIG.5",
        "A-FIG.6",
        "A-FIG.7",
        "B-FIG.1",
        "B-FIG.2",
        "B-FIG.3",
        "B-FIG.4",
        "B-FIG.5",
    ]:
        dfdict[dn] = pd.DataFrame(
            {
                "TreExp": gexp.loc[gexp["Condition"] == dn, "Tre"],
                "TreSim": gsim.loc[gsim["Condition"] == dn, "t_core_pelvis"],
                "TskExp": gexp.loc[gexp["Condition"] == dn, "Tsk"],
                "TskSim": gsim.loc[gsim["Condition"] == dn, "t_skin_mean"],
            },
        )

    dfs = []
    for key, value in dfdict.items():
        for seg in ["Tre", "Tsk"]:
            df = value[[seg + "Exp", seg + "Sim"]].copy()
            df.columns = ["Exp", "Sim"]
            df["Time"] = [i for i in range(0, 241, 5)]
            df["ExpName"] = key
            df["Type"] = seg
            dfs.append(df.copy())

    def plotdata(ser, ax, color, markersize=7, legend=False):
        mk = None
        me = 1
        if color == "black":
            ser.plot(
                ax=ax,
                linewidth=2,
                color=(0.0, 0.0, 0.0, 0.7),
                marker="o",
                markersize=markersize,
                legend=legend,
                markerfacecolor=mk,
                markeredgewidth=me,
            )
        elif color == "blue":
            ser.plot(
                ax=ax,
                linewidth=2,
                color=(0.1, 0.3, 1.0, 0.7),
                marker="^",
                markersize=markersize,
                legend=legend,
                markerfacecolor=mk,
                markeredgewidth=me,
            )
        elif color == "green":
            ser.plot(
                ax=ax,
                linewidth=2,
                color=(0.2, 0.6, 0.2, 0.7),
                marker="s",
                markersize=markersize,
                legend=legend,
                markerfacecolor=mk,
                markeredgewidth=me,
            )
        elif color == "gray":
            ser.plot(
                ax=ax,
                linewidth=2,
                color=(0.3, 0.3, 0.3, 0.7),
                marker="o",
                markersize=markersize,
                legend=legend,
                markerfacecolor=mk,
                markeredgewidth=me,
            )
        elif color == "lightblue":
            ser.plot(
                ax=ax,
                linewidth=2,
                color=(0.3, 0.1, 0.7, 0.7),
                marker="^",
                markersize=markersize,
                legend=legend,
                markerfacecolor=mk,
                markeredgewidth=me,
            )

    def graph(gexp, gsim, ax):
        gexp.index = [i for i in range(0, 241, 5)]
        gsim.index = [i for i in range(0, 241, 5)]

        # Tre
        ser = gexp["Tre"]
        ser.name = "Rectum (EXP)"
        plotdata(ser, ax, "black")
        ser = gsim["t_core_pelvis"]
        ser.name = "Rectum (JOS-3)"
        plotdata(ser, ax, "blue")

        # Tsk
        ser = gexp["Tsk"]
        ser.name = "Mean skin (EXP)"
        plotdata(ser, ax, "gray")
        ser = gsim["t_skin_mean"]
        ser.name = "Mean skin (JOS-3)"
        plotdata(ser, ax, "lightblue")

        ax.set_ylim((28, 40))
        ax.set_yticks([i for i in range(28, 41, 2)])
        ax.set_xlim((0, 240))
        ax.set_xticks([i for i in range(0, 241, 30)])

    inputcons = {
        "A-FIG.4": {"to": [28.0, 27.8, 33.3, 28.0], "rh": [40, 37, 34, 37]},
        "A-FIG.5": {"to": [28.0, 28.5, 37.5, 28.5], "rh": [40, 40, 34, 41]},
        "A-FIG.6": {"to": [28.0, 28.0, 42.5, 28.1], "rh": [40, 37, 34, 37]},
        "A-FIG.7": {"to": [28.0, 28.1, 47.8, 28.3], "rh": [40, 43, 27, 44]},
        "B-FIG.1": {"to": [28.0, 29.0, 22.0, 29.0], "rh": [40, 44, 39, 41]},
        "B-FIG.2": {"to": [28.0, 28.0, 18.0, 28.0], "rh": [40, 40, 40, 40]},
        "B-FIG.3": {"to": [28.0, 22.3, 43.5, 22.6], "rh": [40, 40, 38, 36]},
        "B-FIG.4": {"to": [28.0, 18.0, 42.0, 18.0], "rh": [40, 40, 40, 40]},
        "B-FIG.5": {"to": [28.0, 43.0, 17.0, 43.0], "rh": [40, 40, 40, 40]},
    }

    gexp = exp_dataset["Stolwijk1966"].copy()
    gsim = sim_dataset["Stolwijk1966"].copy()
    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(7, 3),
    )

    for i, dn in enumerate(
        [
            "A-FIG.4",
            "A-FIG.5",
            "A-FIG.6",
            "A-FIG.7",
            "B-FIG.1",
            "B-FIG.2",
            "B-FIG.3",
            "B-FIG.4",
            "B-FIG.5",
        ],
    ):
        # if i >= 4:
        i += 1
        yi = i % 5
        xi = i // 5
        ax = axes[yi, xi]
        graph(gexp.loc[gexp["Condition"] == dn], gsim.loc[gsim["Condition"] == dn], ax)

        ax.set_title(dn)

        ax.set_xticks([i for i in range(0, 241, 30)])
        ax.set_xticklabels([str(i) for i in range(0, 241, 30)])

        ax.axvspan(60, 180, color=(0, 0, 0, 0.1))
        ax.axhspan(39, 40, color=(1, 1, 1, 0.6))
        t = "{:.1f}°C".format(inputcons[dn]["to"][1])
        ax.text(30, 39.1, t, horizontalalignment="center", c="black", fontsize=10)
        t = "{:.1f}°C".format(inputcons[dn]["to"][2])
        ax.text(120, 39.1, t, horizontalalignment="center", c="black", fontsize=10)
        t = "{:.1f}°C".format(inputcons[dn]["to"][3])
        ax.text(210, 39.1, t, horizontalalignment="center", c="black", fontsize=10)

    fig.delaxes(axes[0, 0])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.1, 0.95),
        frameon=False,
        fontsize=12,
    )

    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4)

    fig.text(
        0.03,
        0.5,
        "Body temperature [°C]",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.text(0.53, 0.03, "Time [min]", ha="center", fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    fig.savefig(
        os.path.join(jos3_example_directory, "jos3_validation_with_Stolwijk1966.png"),
    )


def main():
    """Run the selected simulation."""
    # Simulation switch (Please change as needed)
    run_simple_simulation = True
    run_detailed_simulation = True
    run_female_simulation = True
    run_validation_simulation = True

    if run_simple_simulation:
        simple_simulation()
    if run_detailed_simulation:
        detailed_simulation()
    if run_female_simulation:
        female_simulation()
    if run_validation_simulation:
        validation_simulation()


if __name__ == "__main__":
    main()
