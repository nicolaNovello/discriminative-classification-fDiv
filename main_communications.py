from utils import *
from classes import *

data_params = {
        'test_size': 30000,
        'noise_model': "AWGN",  # "PAM_attenuation" "PAM_triangular" "AWGN"
        'SNR_vec': range(-15, 20)
    }
proc_params = {
    'alpha': 0.8,
    'batch_size': 128,
    'lr': 0.001,
    'num_epochs': 500,  # 500 for PAM attenuation, 1000 for PAM triangular
    'cost_function_v': 5,
    'random_seed': 0
}

ber_total = []
ber_total_genie = []
H_x_total = []
H_x_y_total = []
MI_total = []
P_error = []
ber_total_LL = []
ber_total_LL_Mid = []
ber_total_LL_pdf_Mid = []
ber_total_LL_AWGN = []
ber_total_MAP = []

for j, SNR in enumerate(data_params['SNR_vec']):
    print("SNR: ", SNR)
    eps = np.sqrt(pow(10, -0.1 * SNR) / (2 * 0.5))
    if data_params['noise_model'] == "AWGN":
        print("AWGN")
        latent_dim = 6
        output_dim = 2 ** latent_dim
        model = Discriminator(input_dim=latent_dim, output_dim=output_dim)
        combined = CombinedArchitecture(model, cost_function_v=proc_params['cost_function_v'])
        trained_combined = train_communication_awgn(combined, latent_dim, eps=eps, lr=proc_params['lr'],
                                                    num_epochs=proc_params['num_epochs'],
                                                    batch_size=proc_params['batch_size'],
                                                    noise_model=data_params['noise_model'],
                                                    cost_function_v=proc_params['cost_function_v'],
                                                    alpha=proc_params['alpha'], random_seed=proc_params['random_seed'])
        ber, ber_maxL, H_x, H_x_y, MI, P_e, y = test_communication_awgn(trained_combined, latent_dim,
                                                                        test_size=data_params['test_size'],
                                                                        noise_model=data_params['noise_model'],
                                                                        eps=eps, cost_function_v=proc_params[
                                                                        'cost_function_v'], random_seed=proc_params['random_seed'])
        ber_total.append(ber)
        ber_total_genie.append(ber_maxL)
        H_x_total.append(H_x)
        H_x_y_total.append(H_x_y)
        MI_total.append(MI)
        P_error.append(P_e)

    elif data_params['noise_model'] == "PAM_attenuation":
        latent_dim = 1
        M = 4
        model = Discriminator(input_dim=latent_dim, output_dim=M)
        combined = CombinedArchitecture(model, cost_function_v=proc_params['cost_function_v'])
        trained_combined = train_communication_pam_attenuation(combined, latent_dim, M, eps=eps,
                                                               lr=proc_params['lr'],
                                                               num_epochs=proc_params['num_epochs'],
                                                               batch_size=32, save_training_loss=True,
                                                               noise_model=data_params['noise_model'],
                                                               cost_function_v=proc_params['cost_function_v'],
                                                               alpha=proc_params['alpha'], random_seed=proc_params['random_seed'])
        cber, cber_LL, cber_LL_AWGN, MI, R = test_communication_pam_attenuation(
            trained_combined, latent_dim, M,
            test_size=data_params['test_size'],
            noise_model=data_params['noise_model'], eps=eps,
            cost_function_v=proc_params['cost_function_v'], random_seed=proc_params['random_seed'])
        ber_total.append(np.mean(cber))
        ber_total_LL.append(np.mean(cber_LL))
        ber_total_LL_AWGN.append(np.mean(cber_LL_AWGN))
        MI_total.append(np.mean(MI))

    elif data_params['noise_model'] == "PAM_triangular":
        latent_dim = 1
        M = 4
        model = Discriminator(input_dim=latent_dim, output_dim=M)
        combined = CombinedArchitecture(model, cost_function_v=proc_params['cost_function_v'])
        trained_combined = train_communication_pam_triangular(combined, latent_dim, eps=eps, lr=proc_params['lr'],
                                                              num_epochs=proc_params['num_epochs'],
                                                              batch_size=32, save_training_loss=True,
                                                              noise_model=data_params['noise_model'],
                                                              cost_function_v=proc_params['cost_function_v'],
                                                              alpha=proc_params['alpha'], random_seed=proc_params['random_seed'])
        cber, cber_LL, cber_MAP, fDIME_e, _ = test_communication_pam_triangular(
            trained_combined, latent_dim,
            test_size=data_params['test_size'],
            noise_model=data_params['noise_model'], eps=eps,
            cost_function_v=proc_params['cost_function_v'], random_seed=proc_params['random_seed'])

        ber_total.append(np.mean(cber))
        ber_total_LL.append(np.mean(cber_LL))
        ber_total_MAP.append(np.mean(cber_MAP))
        MI_total.append(np.mean(fDIME_e))

