%% Semana 2 – Implementação do Controlador Principal
clear; clc; close all;

%% 1. Parâmetros da Simulação e do Robô
params.n = 2; % Número de juntas
tspan = [0 20]; % Tempo de simulação em segundos 

% Parâmetros físicos do robô 
m1=5; m2=2; L1=1; L2=0.75; I1=0.2; I2=0.2;
params.p1 = m1*(L1/2)^2 + m2*(L2/2)^2 + I1;
params.p2 = m2*(L2/2)^2 + I2;
params.p3 = m2*L1*(L2/2); % Note: L1/2 not (L1/2)^2 as in paper
params.p4 = m1*(L1/2) + m2*L1; % Note: L1/2 not (L2/2)^2 as in paper
params.p5 = m2*(L2/2); % Note: L2/2 not (L2/2)^2 as in paper

%% 2. Parâmetros do Controlador e da RBF NN
% Ganhos do controlador 
params.k0 = 1;
params.alpha = 10;
params.Gamma = 100;
params.varpi = 0.05;
params.kd = 10;

% Parâmetros da RBF NN 
params.N = 10; % Número de neurônios
params.rho = 10; % Largura das funções de base
input_size = 6; % Tamanho do vetor Z = [e(2); e_dot(2); Xi(2)]
% Centros dos neurônios, distribuídos uniformemente no espaço de entrada
params.mu = 2.5 * (2 * rand(input_size, params.N) - 1);

%% 3. Condições Iniciais
q0 = [1; -1]; % Posição inicial 
q_dot0 = [0; 0]; % Velocidade inicial 
e_int0 = [0; 0]; % Erro integral inicial
% Pesos iniciais da NN começam em zero 
Omega_hat0 = zeros(params.N, params.n);

% Montar o vetor de estado inicial 'x0'
x0 = [q0; q_dot0; e_int0; reshape(Omega_hat0, params.N * params.n, 1)];

%% 4. Executar a Simulação
disp('Simulando o sistema...');
options = odeset('RelTol', 1e-4, 'AbsTol', 1e-6);
[t, x] = ode45(@(t,x) robot_dynamics_v0(t, x, params), tspan, x0, options);
disp('Simulação concluída.');

%% 5. Pós-processamento e Análise dos Resultados
% Extrair estados da matriz de saída 'x'
q = x(:, 1:params.n);
q_dot = x(:, params.n+1:2*params.n);

% Recalcular trajetória desejada e erro para plotagem
qd = zeros(length(t), params.n);
e = zeros(length(t), params.n);
for i = 1:length(t)
    qd(i,:) = [sin(t(i)); cos(t(i))];
    e(i,:) = qd(i,:) - q(i,:);
end

%% 6. Plotar Resultados
% Comparar resultados com as figuras do artigo

% Figura de Seguimento da Trajetória
figure('Name', 'Seguimento da Trajetória');
subplot(2,1,1);
plot(t, qd(:,1), 'r--', 'LineWidth', 1.5); hold on;
plot(t, q(:,1), 'b-', 'LineWidth', 1.5);
title('Junta 1: Posição Real vs. Desejada');
xlabel('Tempo (s)'); ylabel('Posição (rad)');
legend('Desejada', 'Real'); grid on;

subplot(2,1,2);
plot(t, qd(:,2), 'r--', 'LineWidth', 1.5); hold on;
plot(t, q(:,2), 'b-', 'LineWidth', 1.5);
title('Junta 2: Posição Real vs. Desejada');
xlabel('Tempo (s)'); ylabel('Posição (rad)');
legend('Desejada', 'Real'); grid on;

% Figura do Erro de Rastreamento (similar à Fig. 2 do artigo)
figure('Name', 'Erro de Rastreamento');
plot(t, e(:,1), 'b--', 'LineWidth', 1.5); hold on;
plot(t, e(:,2), 'r-', 'LineWidth', 1.5);
title('Erro de Rastreamento ao Longo do Tempo');
xlabel('Tempo (s)'); ylabel('Erro (rad)');
legend('Erro Junta 1 (e_1)', 'Erro Junta 2 (e_2)');
grid on;
ylim([-0.05 0.05]); % Ajuste o limite para melhor visualização

disp('Verificação de estabilidade: Observe se os erros convergem para perto de zero.');