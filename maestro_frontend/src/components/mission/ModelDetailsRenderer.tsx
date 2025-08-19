import React from 'react';
import { Cpu, Zap, ChevronsRight } from 'lucide-react';
import type { ExecutionLogEntry } from './AgentActivityLog';

interface ModelDetailsRendererProps {
  log: ExecutionLogEntry;
}

export const ModelDetailsRenderer: React.FC<ModelDetailsRendererProps> = ({ log }) => {
  if (!log.model_details) return null;

  const {
    provider,
    model_name,
    duration_sec,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    cost,
  } = log.model_details;

  return (
    <div className="mt-2 p-2 bg-gray-50 rounded">
      <div className="flex items-center text-xs text-gray-500 mb-2">
        <Cpu className="h-4 w-4 mr-2" />
        <span className="font-semibold">{provider}</span>
        <ChevronsRight className="h-4 w-4 mx-1" />
        <span>{model_name}</span>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-600">
        {duration_sec && (
          <div className="flex items-center">
            <Zap className="h-3 w-3 mr-1" />
            <span>{duration_sec.toFixed(2)}s</span>
          </div>
        )}
        {prompt_tokens && (
          <div>
            <span className="font-medium">P:</span> {prompt_tokens}
          </div>
        )}
        {completion_tokens && (
          <div>
            <span className="font-medium">C:</span> {completion_tokens}
          </div>
        )}
        {total_tokens && (
          <div>
            <span className="font-medium">T:</span> {total_tokens}
          </div>
        )}
        {cost && (
          <div>
            <span className="font-medium">Cost:</span> ${cost.toFixed(6)}
          </div>
        )}
      </div>
    </div>
  );
};
