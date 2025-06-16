clear; clc; close all;

%% 1. Parâmetros da Simulação e do Robô
disp('Inicializando parâmetros...');
params.n = 2; % Número de juntas
tspan = [0 20]; % Tempo de simulação em segundos

% Parâmetros físicos do robô (conforme Seção 5 do artigo)
m1=5; m2=2; L1=1; L2=0.75; I1=0.2; I2=0.2; g=9.81;
params.p1 = m1*(L1/2)^2 + m2*(L1)^2 + m2*(L2/2)^2 + I1 + I2; % Correção baseada em modelos padrão
params.p2 = m2*(L2/2)^2 + I2;
params.p3 = m2*L1*(L2/2);
params.p4 = m1*(L1/2) + m2*L1;
params.p5 = m2*(L2/2);
params.g = g;

%% 2. Parâmetros do Controlador e da RBF NN
% Ganhos e parâmetros do controlador (conforme Seção 5 do artigo)
params.k0 = 1;
params.alpha = 10;
params.Gamma = 100;
params.varpi = 0.05;
params.kd = 10;
params.kb = 1;      % Ganho da barreira
params.C = 0.1;     % Limite da restrição |Ξ| < C

% Parâmetros da RBF NN
params.N = 10; % Número de neurônios
params.rho = 10; % Largura das funções de base
input_size = 6; % Tamanho do vetor Z = [e(2); e_dot(2); Xi(2)]
% Centros dos neurônios, distribuídos no intervalo [-2.5, 2.5]
params.mu = 2.5 * (2 * rand(input_size, params.N) - 1);

%% 3. Condições Iniciais
% Conforme Seção 5 do artigo
q0 = [1; -1];
q_dot0 = [0; 0];
% Condições iniciais para os estados do integrador e da NN
e_int0 = [0; 0];
Omega_hat0 = zeros(params.N, params.n);

% Montar o vetor de estado inicial 'x0'
x0 = [q0; q_dot0; e_int0; reshape(Omega_hat0, params.N * params.n, 1)];

%% 4. Executar a Simulação com o Solver ODE45
disp('Simulando o sistema... Isso pode levar alguns segundos.');
options = odeset('RelTol', 1e-4, 'AbsTol', 1e-6, 'Stats', 'on');
[t, x] = ode45(@(t,x) robot_dynamics(t, x, params), tspan, x0, options);
disp('Simulação concluída.');

%% 5. Pós-processamento para Geração de Gráficos
disp('Processando resultados para plotagem...');
n_steps = length(t);
q_hist = x(:, 1:params.n);
Omega_hat_hist = x(:, 3*params.n+1:end);

% Inicializar vetores para armazenar variáveis
e_hist = zeros(n_steps, params.n);
Xi_hist = zeros(n_steps, params.n);
tau_hist = zeros(n_steps, params.n);
kappa_D_hist = zeros(n_steps, params.n);
tau_blf_hist = zeros(n_steps, params.n);
qd_hist = zeros(n_steps, params.n);

% Loop para recalcular variáveis em cada passo de tempo
for i = 1:n_steps
    xi = x(i,:)';
    ti = t(i);
    
    % Desempacotar estados (igual ao início de robot_dynamics)
    qi = xi(1:params.n);
    q_doti = xi(params.n+1:2*params.n);
    e_inti = xi(2*params.n+1:3*params.n);
    Omega_hati = reshape(xi(3*params.n+1:end), params.N, params.n); 
    
    qdi = [cos(ti); -cos(ti)]; % Trajetória do artigo
    qd_doti = [-sin(ti); sin(ti)];
    
    ei = qdi - qi;
    e_doti = qd_doti - q_doti;
    Xi_i = 2*params.k0*ei + params.k0^2*e_inti + e_doti;
    
    Zi = [ei; e_doti; Xi_i];
    hi = zeros(params.N, 1);
    for j = 1:params.N, hi(j) = exp(-norm(Zi - params.mu(:, j))^2 / (2 * params.rho^2)); end
    
    kappa_Di = params.alpha * (Omega_hati' * hi);
    K_dblfi = params.kb ./ (params.C^2 - Xi_i.^2 + 1e-9);
    K_Dblf_totali = params.kd + kappa_Di + K_dblfi;
    tau_i = K_Dblf_totali .* Xi_i;
    tau_blfi = K_dblfi .* Xi_i;
    
    qd_hist(i,:) = qdi';
    e_hist(i,:) = ei';
    Xi_hist(i,:) = Xi_i';
    tau_hist(i,:) = tau_i';
    kappa_D_hist(i,:) = kappa_Di';
    tau_blf_hist(i,:) = tau_blfi';
end

%% 6. Plotar Resultados
disp('Graficos vem ai...');

% Figura de Seguimento da Trajetória
figure('Name', 'Seguimento da Trajetória');
subplot(2,1,1);
plot(t, qd_hist(:,1), 'r--', 'LineWidth', 1.5); hold on;
plot(t, q_hist(:,1), 'b-', 'LineWidth', 1.5);
title('Junta 1: Posição Real vs. Desejada');
xlabel('Tempo (s)'); ylabel('Posição (rad)');
legend('Desejada', 'Real'); grid on;
subplot(2,1,2);
plot(t, qd_hist(:,2), 'r--', 'LineWidth', 1.5); hold on;
plot(t, q_hist(:,2), 'b-', 'LineWidth', 1.5);
title('Junta 2: Posição Real vs. Desejada');
xlabel('Tempo (s)'); ylabel('Posição (rad)');
legend('Desejada', 'Real'); grid on;

% Erro de Rastreamento (similar à Fig. 2 e 8)
figure('Name','Erro de Rastreamento');
plot(t, e_hist); title('Erro de Rastreamento e(t)');
xlabel('Tempo (s)'); ylabel('Erro (rad)'); legend('e_1', 'e_2'); grid on;

% Erro Filtrado e Restrições (similar à Fig. 4)
figure('Name','Erro Filtrado Ξ e Restrições');
plot(t, Xi_hist(:,1), 'b--', 'LineWidth', 1.5); hold on;
plot(t, Xi_hist(:,2), 'r-', 'LineWidth', 1.5);
plot(t, ones(size(t))*params.C, 'k-.', 'LineWidth', 1);
plot(t, -ones(size(t))*params.C, 'k-.', 'LineWidth', 1);
title('Erro Filtrado Ξ(t) e Limites de Restrição');
xlabel('Tempo (s)'); ylabel('Ξ');
legend('Ξ_1', 'Ξ_2', 'Limite C', 'Limite -C'); grid on;
ylim([-params.C*1.5, params.C*1.5]);

% Ganho Adaptativo do PID (similar à Fig. 5)
figure('Name','Ganho Adaptativo κ_D');
plot(t, kappa_D_hist); title('Ganho Adaptativo κ_D(t)');
xlabel('Tempo (s)'); ylabel('Ganho'); legend('κ_{D1}', 'κ_{D2}'); grid on;

% Pesos da NN (similar à Fig. 6)
figure('Name','Pesos da Rede Neural');
subplot(1,2,1); plot(t, Omega_hat_hist(:, 1:params.N)); title('Pesos da NN para Junta 1');
xlabel('Tempo (s)'); ylabel('$\hat{\Omega}_1$','Interpreter','latex'); grid on;
subplot(1,2,2); plot(t, Omega_hat_hist(:, params.N+1:end)); title('Pesos da NN para Junta 2');
xlabel('Tempo (s)'); ylabel('$\hat{\Omega}_2$','Interpreter','latex'); grid on;

% Torque de Controle Total (similar à Fig. 7)
figure('Name','Torque de Controle');
plot(t, tau_hist); title('Torque de Controle Total τ(t)');
xlabel('Tempo (s)'); ylabel('Torque (Nm)'); legend('τ_1', 'τ_2'); grid on;

% Torque da BLF (similar à Fig. 11)
figure('Name','Torque da BLF');
plot(t, tau_blf_hist); title('Torque Gerado pela BLF τ_{blf}(t)');
xlabel('Tempo (s)'); ylabel('Torque (Nm)'); legend('τ_{blf1}', 'τ_{blf2}'); grid on;