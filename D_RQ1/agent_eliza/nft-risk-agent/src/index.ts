import { logger, type IAgentRuntime, type Project, type ProjectAgent } from '@elizaos/core';
import nftRiskPlugin from './plugin.ts';
import { character } from './character.ts';

const initCharacter = ({ runtime }: { runtime: IAgentRuntime }) => {
  logger.info('Initializing character');
  logger.info({ name: character.name }, 'Name:');
};

export const projectAgent: ProjectAgent = {
  character,
  init: async (runtime: IAgentRuntime) => await initCharacter({ runtime }),
  plugins: [nftRiskPlugin],
};

const project: Project = {
  agents: [projectAgent],
};

export { character } from './character.ts';

export default project;
