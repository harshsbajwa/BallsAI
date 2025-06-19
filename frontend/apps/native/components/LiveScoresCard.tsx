import React from "react";
import { View, Text, ScrollView } from "react-native";
import { Loading } from "./Loading";
import { Game } from "@nba/api/types";

interface LiveScoresCardProps {
  games?: Game[];
  loading?: boolean;
}

export function LiveScoresCard({ games = [], loading }: LiveScoresCardProps) {
  return (
    <View className="bg-white rounded-lg shadow-sm p-4">
      <Text className="text-lg font-bold text-gray-900 mb-4">
        Today's Games
      </Text>

      {loading ? (
        <Loading text="Loading games..." />
      ) : games.length === 0 ? (
        <Text className="text-gray-500 text-center py-8">
          No games scheduled today
        </Text>
      ) : (
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <View className="flex-row space-x-4">
            {games.map((game) => (
              <View
                key={game.game_id}
                className="bg-gray-50 rounded-lg p-3 min-w-[180px] border border-gray-200"
              >
                <View className="flex-row justify-between items-center">
                  <View className="flex-1 space-y-1">
                    <Text className="font-semibold text-sm">
                      {game.away_team.team_abbrev}
                    </Text>
                    <Text className="font-semibold text-sm">
                      {game.home_team.team_abbrev}
                    </Text>
                  </View>
                  <View className="items-end">
                    {game.away_score !== null && game.home_score !== null ? (
                      <>
                        <Text className="font-bold text-lg">
                          {game.away_score}
                        </Text>
                        <Text className="font-bold text-lg">
                          {game.home_score}
                        </Text>
                      </>
                    ) : (
                      <Text className="text-gray-500 text-sm">
                        {new Date(game.game_date).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </Text>
                    )}
                  </View>
                </View>
              </View>
            ))}
          </View>
        </ScrollView>
      )}
    </View>
  );
}