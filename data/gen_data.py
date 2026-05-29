import numpy as np
import sys
import pandas as pd

## Configuration parameters

G = 1.0          # gravitational constant
DT = 0.001        # timestep
STEPS = 10     # number of simulation steps
SOFTENING = 0.05 # avoids singularities at tiny distances

num_sets = 500

def generate_positions(N, masses, Rmax=1.0, r_min=0.05):
    while True:

        # sample in disk
        r = np.sqrt(np.random.uniform(0, Rmax**2, N))
        theta = np.random.uniform(0, 2*np.pi, N)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        positions = np.column_stack((x, y))

        # check pairwise distances
        valid = True

        for i in range(N):
            for j in range(i + 1, N):

                dist = np.linalg.norm(
                    positions[i] - positions[j]
                )

                if dist < r_min:
                    valid = False
                    break

            if not valid:
                break

        if valid:
             # center-of-mass shift
            com_pos = np.average(
                positions,
                axis=0,
                weights=masses
            )

            positions -= com_pos

            return positions
        
def generate_masses(N):
    masses = 10 ** np.random.uniform(-2, 2, size=N)
    masses /= masses.sum()
    return masses

def generate_velocities(N, masses, positions):
    # Initially Random Velocities
    velocities = np.random.normal(
        size=(N, 2)
    )
    # remove COM drift
    com_vel = np.average(velocities, axis=0, weights=masses)
    velocities -= com_vel

    # Potential Energy
    U = 0.0
    softening = 1e-2

    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[j] - positions[i]
            dist = np.sqrt(np.sum(dx**2) + softening**2)

            U -= G * masses[i] * masses[j] / dist

    # Kinetic Energy
    speed_sq = np.sum(velocities**2, axis=1)
    T = 0.5 * np.sum(masses * speed_sq)

    # safety check
    if T < 1e-12:
        velocities += 1e-3 * np.random.normal(size=velocities.shape)

        speed_sq = np.sum(velocities**2, axis=1)

        T = 0.5 * np.sum(masses * speed_sq,)

    #Target virial ration Q
    Q_target = np.random.uniform(0.2, 1.2)

    T_target = Q_target * abs(U)

    # scale velocities
    scale = np.sqrt(T_target / T)
    velocities *= scale

    # Finally
    T_final = 0.5 * np.sum(
        masses * np.sum(velocities**2, axis=1)
    )

    Q_final = T_final / abs(U)

    print("Q_target =", Q_target)
    print("Q_final  =", Q_final)
    
    return velocities, Q_final

N_list = np.random.randint(2,10, size = num_sets)
records = []

for sim_id, N in enumerate(N_list):
    print(f"Working for {N} bodies")
    masses = generate_masses(N)
    
    positions = generate_positions(N, masses)
    #print(f"Positions = {positions}")
    velocities, Q_final = generate_velocities(N, masses, positions)

    trajectory = np.zeros((STEPS, N, 2))
    
    filename = f"ic_{sim_id:06d}.npz"
    np.savez_compressed(
        f"initial_conditions/{filename}",
        masses=masses,
        positions=positions,
        velocities=velocities,
        num_bodies=N,
        virial_ratio=Q_final 
    )

    # append one record to the list
    records.append({
        "sim_id": sim_id,
        "filename": filename,
        "N": int(N),
        "virial_ratio": round(Q_final, 4),
        "total_mass": round(float(masses.sum()), 4),
        "mass_ratio_max": round(float(masses.max() / masses.min()), 4)
    })

# save the index once, after the loop
df = pd.DataFrame(records)
df.to_csv("initial_conditions/index.csv", index=False)
