import numpy as np
import pygame
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.conversions import aec_to_parallel

# --- ENVIRONNEMENT ---
class CustomRenderEnv(SimpleEnv):
    
    def _execute_world_step(self):
        super()._execute_world_step()
        self.resolve_collisions() 
        self.enforce_boundaries() 
        # Pas de check_goal_snap()

    # 1. OBSTACLES (Inchangé)
    def resolve_collisions(self):
        for agent in self.world.agents:
            for lm in self.world.landmarks:
                if 'obstacle' in lm.name:
                    ox, oy = lm.state.p_pos; half_s = lm.size
                    ax, ay = agent.state.p_pos
                    closest_x = max(ox - half_s, min(ax, ox + half_s))
                    closest_y = max(oy - half_s, min(ay, oy + half_s))
                    diff_x = ax - closest_x; diff_y = ay - closest_y
                    dist = np.sqrt(diff_x**2 + diff_y**2)
                    
                    if dist < agent.size:
                        if dist == 0: diff_x, diff_y, dist = 1.0, 0.0, 1.0
                        overlap = agent.size - dist
                        agent.state.p_pos[0] += (diff_x / dist) * overlap
                        agent.state.p_pos[1] += (diff_y / dist) * overlap
                        agent.state.p_vel[:] = 0.0

    # 2. LIMITES DU MONDE (MODIFIÉ) 
    def enforce_boundaries(self):
        wall_limit = 1.0
        
        for agent in self.world.agents:
            # La limite effective pour le CENTRE est (Mur - Rayon)
            # Ainsi, quand le centre est à cette limite, le bord touche le mur.
            effective_limit = wall_limit - agent.size
            
            # Check X (Gauche / Droite)
            if agent.state.p_pos[0] > effective_limit:
                agent.state.p_pos[0] = effective_limit
                agent.state.p_vel[0] = 0.0 # Stop net
            elif agent.state.p_pos[0] < -effective_limit:
                agent.state.p_pos[0] = -effective_limit
                agent.state.p_vel[0] = 0.0
                
            # Check Y (Bas / Haut)
            if agent.state.p_pos[1] > effective_limit:
                agent.state.p_pos[1] = effective_limit
                agent.state.p_vel[1] = 0.0
            elif agent.state.p_pos[1] < -effective_limit:
                agent.state.p_pos[1] = -effective_limit
                agent.state.p_vel[1] = 0.0

    def step(self, action):
        super().step(action)
        for agent in self.world.agents:
            if self.scenario.is_done(agent, self.world):
                self.terminations[agent.name] = True

    # --- RENDU ---
    def render(self):
        if self.render_mode is None: return
        if self.render_mode == "human":
            if pygame.display.get_surface() is None:
                pygame.init(); pygame.display.set_caption("MARL - Hard Boundaries")
                self.screen = pygame.display.set_mode((700, 700))
            elif self.screen is None: self.screen = pygame.display.get_surface()     
        elif self.render_mode == "rgb_array":
            if self.screen is None: pygame.init(); self.screen = pygame.Surface((700, 700))

        width, height = self.screen.get_size(); scale = width / 2.0 
        def to_screen(pos): return int((pos[0] + 1) * (width / 2)), int(height - ((pos[1] + 1) * (height / 2)))

        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (0,0,0), (0,0,width,height), 5)

        obs_radius = getattr(self.scenario, 'obs_radius', 0.5)
        entities = sorted(self.world.entities, key=lambda e: 0 if isinstance(e, Landmark) else 1)

        for entity in entities:
            color = (int(entity.color[0] * 255), int(entity.color[1] * 255), int(entity.color[2] * 255))
            pos_px = to_screen(entity.state.p_pos); size_px = int(entity.size * scale)

            if 'target' in entity.name or 'obstacle' in entity.name:
                rect = pygame.Rect(pos_px[0]-size_px, pos_px[1]-size_px, size_px*2, size_px*2)
                if 'target' in entity.name: 
                    pygame.draw.rect(self.screen, color, rect) 
                    pygame.draw.circle(self.screen, (0,0,0), pos_px, 2)
                else:
                    pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0,0,0), rect, 2)
            else:
                # Visibilité
                viz_radius = int(obs_radius * scale)
                pygame.draw.circle(self.screen, (220, 220, 220), pos_px, viz_radius, 1)
                
                # Agent
                pygame.draw.circle(self.screen, color, pos_px, size_px)
                pygame.draw.circle(self.screen, (0,0,0), pos_px, size_px, 1)
                pygame.draw.circle(self.screen, (0,0,0), pos_px, 2) 

        if self.render_mode == "human": pygame.display.flip()
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

# --- SCENARIO ---
class Scenario(BaseScenario):
    def make_world(self, config):
        world = World()
        world.dim_c = 0; world.dim_p = 2; world.damping = config['world'].get('damping', 0.25)
        self.obs_radius = config['world'].get('obs_radius', 0.5)
        self.max_agents = config['world'].get('total_agents_possible', 1)
        world.landmarks = [] 
        self.goal_map = {} 

        # Obstacles
        for i, obs_cfg in enumerate(config.get('obstacles', [])):
            lm = Landmark(); lm.name = f'obstacle_{i}'; lm.collide = True; lm.movable = False; lm.size = obs_cfg['size']
            lm.state.p_pos = np.array(obs_cfg['pos']); lm.state.p_vel = np.zeros(world.dim_p)
            lm.color = np.array([0.2, 0.2, 0.2]); world.landmarks.append(lm)

        # Goals
        for agent_cfg in config['agents']:
            goal_lm = Landmark(); goal_lm.name = f'target_{agent_cfg["id"]}'; goal_lm.collide = False; goal_lm.movable = False
            
            # Goal Size = Agent Size * 1.5
            agent_size = agent_cfg.get('size', 0.1)
            goal_lm.size = agent_size * 1.5 
            
            goal_lm.state.p_pos = np.array(agent_cfg['goal'])
            goal_lm.state.p_vel = np.zeros(world.dim_p)
            goal_lm.color = np.array(agent_cfg.get('color', [0.5, 0.5, 0.5]))
            goal_lm.color = np.clip(goal_lm.color + 0.2, 0, 1) 
            
            world.landmarks.append(goal_lm)
            self.goal_map[f'agent_{agent_cfg["id"]}'] = goal_lm

        # Agents
        world.agents = []
        for agent_cfg in config['agents']:
            ag = Agent(); ag.name = f'agent_{agent_cfg["id"]}'; ag.collide = True; ag.silent = True
            ag.size = agent_cfg.get('size', 0.1); ag.accel = agent_cfg.get('accel', 3.0); ag.max_speed = agent_cfg.get('max_speed', 1.0)
            ag.start_pos = np.array(agent_cfg['start']); ag.color = np.array(agent_cfg.get('color', [0.5, 0.5, 0.5]))
            world.agents.append(ag)
        return world

    def reset_world(self, world, np_random):
        for agent in world.agents:
            agent.state.p_pos = agent.start_pos.copy()
            agent.state.p_vel = np.zeros(world.dim_p); agent.state.c = np.zeros(world.dim_c)
        for lm in world.landmarks: lm.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if self.is_done(agent, world): 
            return 100.0
        
        rew = -1.0 

        for lm in world.landmarks:
            if 'obstacle' in lm.name:
                diff = agent.state.p_pos - lm.state.p_pos
                closest = np.clip(diff, -lm.size, lm.size)
                dist_sq = np.sum((diff - closest)**2)
                if dist_sq <= (agent.size ** 2) + 0.0001: 
                    rew -= 50.0 
        
        return rew

    def is_done(self, agent, world):
        goal_lm = self.goal_map.get(agent.name)
        if not goal_lm: return False
        dist = np.linalg.norm(agent.state.p_pos - goal_lm.state.p_pos)
        return dist < 0.05

    def check_visibility(self, observer_pos, target_pos, world):
        dist = np.linalg.norm(observer_pos - target_pos)
        if dist > self.obs_radius: return False 
        A = observer_pos; B = target_pos; AB = B - A; len_AB = np.linalg.norm(AB)
        if len_AB == 0: return True
        for lm in world.landmarks:
            if 'obstacle' in lm.name:
                C = lm.state.p_pos; r = lm.size; AC = C - A; t = np.dot(AC, AB) / (len_AB**2)
                t = max(0, min(1, t)); Closest = A + t * AB
                if np.linalg.norm(C - Closest) < r: return False 
        return True

    def observation(self, agent, world):
        return np.concatenate([agent.state.p_vel, agent.state.p_pos])

def make_env(config, render_mode=None):
    scenario = Scenario(); world = scenario.make_world(config)
    env = CustomRenderEnv(scenario, world, max_cycles=1000, render_mode=render_mode, continuous_actions=True)
    env.unwrapped.scenario = scenario; env = aec_to_parallel(env)
    return env