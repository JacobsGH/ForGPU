import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ParticleSimulation:
    def __init__(self, num_particles, width, height, gravity=9.81, restitution_coefficient=0.8):
        # Настроим параметры
        self.num_particles = num_particles
        self.width = width
        self.height = height
        self.gravity = gravity
        self.restitution_coefficient = restitution_coefficient
        
        # � нициализация OpenCL
        self.platform = cl.get_platforms()[0]  # используем первую платформу
        self.device = self.platform.get_devices()[0]  # используем первое устройство
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
        # Буферы данных для частиц
        self.positions = np.random.rand(num_particles, 2).astype(np.float32) * [width, height]  # случайные позиции
        self.velocities = np.random.rand(num_particles, 2).astype(np.float32) * 10  # случайные скорости
        self.masses = np.ones(num_particles, dtype=np.float32)  # единичные массы
        self.restitution = np.full(num_particles, restitution_coefficient, dtype=np.float32)
        
        # Создание OpenCL буферов
        self.positions_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.positions)
        self.velocities_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.velocities)
        self.masses_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.masses)
        self.restitution_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.restitution)
        
        # OpenCL код
        self.kernel_code = """
        __kernel void update_particles(
            __global float2* positions,
            __global float2* velocities,
            __global float* masses,
            __global float* restitution,
            const float dt,
            const float gravity,
            const float width,
            const float height
        ) {
            int i = get_global_id(0); // � ндекс текущей частицы
            
            // Загрузка текущего состояния
            float2 pos = positions[i];
            float2 vel = velocities[i];
            float mass = masses[i];
            float res = restitution[i];
            
            // Обновление скорости с учетом гравитации
            vel.y -= gravity * dt;
            
            // Обновление позиции
            pos.x += vel.x * dt;
            pos.y += vel.y * dt;
            
            // Обработка столкновений с стенками
            if (pos.x < 0 || pos.x > width) {
                vel.x = -vel.x * res;  // Отражение с потерей энергии
                pos.x = clamp(pos.x, 0.f, width);  // Ограничение позиции
            }
            if (pos.y < 0 || pos.y > height) {
                vel.y = -vel.y * res;  // Отражение с потерей энергии
                pos.y = clamp(pos.y, 0.f, height);  // Ограничение позиции
            }
            
            // Сохранение нового состояния
            positions[i] = pos;
            velocities[i] = vel;
        }
        """
        
        # Компиляция и создание OpenCL кернела
        self.program = cl.Program(self.context, self.kernel_code).build()
        self.kernel = self.program.update_particles
        
    def update(self, dt):
        # Запуск OpenCL кернела
        self.kernel.set_args(self.positions_buffer, self.velocities_buffer, self.masses_buffer,
                             self.restitution_buffer, np.float32(dt), np.float32(self.gravity),
                             np.float32(self.width), np.float32(self.height))
        
        # Выполнение вычислений
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, (self.num_particles,), None).wait()
        
    def get_positions(self):
        # Получаем текущие позиции частиц
        positions = np.empty_like(self.positions)
        cl.enqueue_copy(self.queue, positions, self.positions_buffer)
        return positions

def main():
    # Параметры симуляции
    num_particles = 3
    width, height = 1200, 800
    sim = ParticleSimulation(num_particles, width, height)

    # Настройка визуализации
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], color = 'red')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    

    # Функция обновления для анимации
    def update(frame):
        # Обновляем состояние частиц
        sim.update(dt=0.001)
        
        # Получаем новые позиции для визуализации
    
        positions = sim.get_positions()
        print(positions)
        scatter.set_offsets(positions)
        return scatter,

    # Запуск анимации
    anim = FuncAnimation(fig, update, frames=None, interval=20, blit=True)
    plt.show()

if __name__ == '__main__':
    main()